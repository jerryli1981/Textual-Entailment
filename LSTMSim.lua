local LSTMSim = torch.class('LSTMSim')

function LSTMSim:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.05
  self.batch_size    = config.batch_size    or 25
  self.reg           = config.reg           or 1e-4
  self.sim_nhidden   = config.sim_nhidden   or 50

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb_vecs = config.emb_vecs

  self.num_classes = 3

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- KL divergence optimization objective
  --self.criterion = nn.DistKLDivCriterion()
  self.criterion = nn.ClassNLLCriterion()

  -- initialize lstm model
  local lstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    gate_output = false,
  }
  

  self.llstm = LSTM(lstm_config)
  self.rlstm = LSTM(lstm_config)

  -- similarity model
  self.sim_module = self:new_sim_module_complex()

  local modules = nn.Parallel()
    :add(self.llstm)
    :add(self.sim_module)
  self.params, self.grad_params = modules:getParameters()

  share_params(self.rlstm, self.llstm)
end

function LSTMSim:new_sim_module()
  local vecs_to_input
  local lvec = nn.Identity()()
  local rvec = nn.Identity()()
  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local cosine_dist = nn.CosineDistance(){lvec, rvec}
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  vecs_to_input = nn.gModule({lvec, rvec}, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(2 * self.mem_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module
end

function LSTMSim:new_sim_module_complex()
  local vecs_to_input

  local lmat = nn.Identity()()
  local lmat_s = nn.SplitTable(1)(lmat)
  local rmat = nn.Identity()()
  local rmat_s = nn.SplitTable(1)(rmat)

  --local k = 1
  local sim_mat = {}
  for i=27, 36 do
    local lvec = nn.SelectTable(i)(lmat_s)
    for j=27, 36 do
      local rvec = nn.SelectTable(j)(rmat_s)
      local cosine_dist = nn.CosineDistance(){lvec, rvec}
      table.insert(sim_mat, cosine_dist)
    end
  end

  sim_mat = nn.Identity()(sim_mat)
  sim_mat_j = nn.JoinTable(1){sim_mat}
  sim_mat_r = nn.Reshape(10,10){sim_mat_j}

  vecs_to_input = nn.gModule({lmat, rmat}, {sim_mat_r})

  local inputFrameSize = 10
  local outputFrameSize = 20
  local kernel_width = 3
  local reduced_l = 10 - kernel_width + 1 

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kernel_width))--(36-kw+1, outputFrameSize) 
    :add(nn.Tanh())
    :add(nn.TemporalMaxPooling(reduced_l)) --(1, outputFrameSize)
    :add(nn.Linear(outputFrameSize, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module

end


function LSTMSim:train(dataset)
  self.llstm:training()
  self.rlstm:training()
  local indices = torch.randperm(dataset.size)

  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local feval = function(x)
      self.grad_params:zero()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local ent = dataset.labels[idx]

        seq_len = 36

        local linputs = self.emb_vecs:index(1, lsent:long()):double()
        local rinputs = self.emb_vecs:index(1, rsent:long()):double()

        local inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
        local output = self.sim_module:forward(inputs)

        
        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, ent)

        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, ent)
        local rep_grad = self.sim_module:backward(inputs, sim_grad)

        self.llstm:backward(linputs, rep_grad[1])
        self.rlstm:backward(rinputs, rep_grad[2])

      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
  end
  xlua.progress(dataset.size, dataset.size)
end

-- Predict the similarity of a sentence pair.
function LSTMSim:predict(lsent, rsent)
  self.llstm:evaluate()
  self.rlstm:evaluate()

  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()
  local inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}

  local output = self.sim_module:forward(inputs)
  self.llstm:forget()
  self.rlstm:forget()

  return output
end

-- Produce similarity predictions for each sentence pair in the dataset.
function LSTMSim:predict_dataset(dataset)

  local predictions = torch.Tensor(dataset.size, self.num_classes)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(lsent, rsent)
  end
  return predictions
end

function LSTMSim:print_config()
  local num_params = self.params:nElement()
  local num_sim_params = self:new_sim_module():getParameters():nElement()
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end