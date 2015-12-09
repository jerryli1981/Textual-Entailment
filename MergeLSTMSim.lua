local MergeLSTMSim = torch.class('MergeLSTMSim')

function MergeLSTMSim:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.05
  self.batch_size    = config.batch_size    or 25
  self.reg           = config.reg           or 1e-4
  self.sim_nhidden   = config.sim_nhidden   or 50
  self.structure     = config.structure     or 'lstm'
  self.num_layers    = config.num_layers    or 1
  self.emb_learning_rate = config.emb_learning_rate or 0.1

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  self.num_classes = 3

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  self.criterion = nn.ClassNLLCriterion()

  -- initialize lstm model
  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    num_layers = self.num_layers,
  }
  

  if self.structure == 'lstm' then
    self.lstm = LSTM(lstm_config) -- "left" LSTM
  elseif self.structure == 'bilstm' then
    self.lstm = LSTM(lstm_config)
    self.lstm_b = LSTM(lstm_config) -- backward "left" LSTM
  else
    error('invalid LSTM type: ' .. self.structure)
  end

  self.ent_module = self:new_ent_module()

  local modules = nn.Parallel()
    :add(self.lstm)
    :add(self.ent_module)

  self.params, self.grad_params = modules:getParameters()

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters

  if self.structure == 'bilstm' then
    -- tying the forward and backward weights improves performance
    share_params(self.lstm_b, self.lstm)
  end
end

function MergeLSTMSim:new_ent_module()
  local input_dim = self.num_layers * self.mem_dim
  local inputs, vec
  if self.structure == 'lstm' then
    local rep = nn.Identity()()
    if self.num_layers == 1 then
      vec = {rep}
    else
      vec = nn.JoinTable(1)(rep)
    end
    inputs = {rep}
  elseif self.structure == 'bilstm' then
    local frep, brep = nn.Identity()(), nn.Identity()()
    input_dim = input_dim * 2
    if self.num_layers == 1 then
      vec = nn.JoinTable(1){frep, brep}
    else
      vec = nn.JoinTable(1){nn.JoinTable(1)(frep), nn.JoinTable(1)(brep)}
    end
    inputs = {frep, brep}
  end

  local logprobs
  if self.dropout then
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes)(
        nn.Dropout()(vec)))
  else
    logprobs = nn.LogSoftMax()(
      nn.Linear(input_dim, self.num_classes)(vec))
  end

  return nn.gModule(inputs, {logprobs})
end

function MergeLSTMSim:train(dataset)
  self.lstm:training()
  self.ent_module:training()
  if self.structure == 'bilstm' then
    self.lstm_b:training()
  end

  local indices = torch.randperm(dataset.size)
  local zeros = torch.zeros(self.mem_dim)
  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()
      self.emb:zeroGradParameters()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local tree = dataset.mtrees[idx]
        local sent = dataset.msents[idx]
        local ent = dataset.labels[idx]

        local inputs = self.emb:forward(sent)

        -- get sentence representations
        local rep
        if self.structure == 'lstm' then
          rep = self.lstm:forward(inputs)
        elseif self.structure == 'bilstm' then
          rep = {
            self.lstm:forward(inputs),
            self.lstm_b:forward(inputs, true), -- true => reverse
          }
        end

        -- compute class log probabilities
        local output = self.ent_module:forward(rep)

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, ent)
        loss = loss + example_loss
        local obj_grad = self.criterion:backward(output, ent)
        local rep_grad = self.ent_module:backward(rep, obj_grad)
        local input_grads
        if self.structure == 'lstm' then
          input_grads = self:LSTM_backward(sent, inputs, rep_grad)
        elseif self.structure == 'bilstm' then
          input_grads = self:BiLSTM_backward(sent, inputs, rep_grad)
        end
        self.emb:backward(sent, input_grads)
      
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
    self.emb:updateParameters(self.emb_learning_rate)
  end
  xlua.progress(dataset.size, dataset.size)
end

-- LSTM backward propagation
function MergeLSTMSim:LSTM_backward(sent, inputs, rep_grad)
  local grad
  if self.num_layers == 1 then
    grad = torch.zeros(sent:nElement(), self.mem_dim)
    grad[sent:nElement()] = rep_grad
  else
    grad = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      grad[{sent:nElement(), l, {}}] = rep_grad[l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad)
  return input_grads
end

-- Bidirectional LSTM backward propagation
function MergeLSTMSim:BiLSTM_backward(sent, inputs, rep_grad)
  local grad, grad_b
  if self.num_layers == 1 then
    grad   = torch.zeros(sent:nElement(), self.mem_dim)
    grad_b = torch.zeros(sent:nElement(), self.mem_dim)
    grad[sent:nElement()] = rep_grad[1]
    grad_b[1] = rep_grad[2]
  else
    grad   = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    grad_b = torch.zeros(sent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      grad[{sent:nElement(), l, {}}] = rep_grad[1][l]
      grad_b[{1, l, {}}] = rep_grad[2][l]
    end
  end
  local input_grads = self.lstm:backward(inputs, grad)
  local input_grads_b = self.lstm_b:backward(inputs, grad_b, true)
  return input_grads + input_grads_b
end

-- Predict the sentiment of a sentence.
function MergeLSTMSim:predict(sent)
  self.lstm:evaluate()
  self.ent_module:evaluate()
  local inputs = self.emb:forward(sent)

  local rep
  if self.structure == 'lstm' then
    rep = self.lstm:forward(inputs)
  elseif self.structure == 'bilstm' then
    self.lstm_b:evaluate()
    rep = {
      self.lstm:forward(inputs),
      self.lstm_b:forward(inputs, true),
    }
  end
  local logprobs = self.ent_module:forward(rep)
  local prediction

  prediction = argmax(logprobs)

  self.lstm:forget()
  if self.structure == 'bilstm' then
    self.lstm_b:forget()
  end
  return prediction
end

-- Produce sentiment predictions for each sentence in the dataset.
function MergeLSTMSim:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = self:predict(dataset.msents[i])
  end
  return predictions
end

function argmax(v)
  local idx = 1
  local max = v[1]
  for i = 2, v:size(1) do
    if v[i] > max then
      max = v[i]
      idx = i
    end
  end
  return idx
end

function MergeLSTMSim:print_config()
  --local num_params = self.params:nElement()
  --local num_sim_params = self:new_sim_module():getParameters():nElement()
  --printf('%-25s = %d\n',   'num params', num_params)
  --printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %s\n',   'LSTM structure', self.structure)
  printf('%-25s = %d\n',   'LSTM layers', self.num_layers)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end