local MergeTreeLSTMSim = torch.class('MergeTreeLSTMSim')

function MergeTreeLSTMSim:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.05
  self.emb_learning_rate = config.emb_learning_rate or 0.0
  self.batch_size    = config.batch_size    or 25
  self.reg           = config.reg           or 1e-4
  self.sim_nhidden   = config.sim_nhidden   or 50

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb = nn.LookupTable(config.emb_vecs:size(1), self.emb_dim)
  self.emb.weight:copy(config.emb_vecs)

  self.num_classes = 3

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate }

  -- KL divergence optimization objective
  --self.criterion = nn.DistKLDivCriterion()
  self.criterion = nn.ClassNLLCriterion()

  -- initialize tree-lstm model
  local treelstm_config = {
    in_dim = self.emb_dim,
    mem_dim = self.mem_dim,
    output_module_fn = function() return self:new_sim_module() end,
    criterion = self.criterion,
  }
  

  self.treelstm = ChildSumTreeLSTM(treelstm_config)

  self.params, self.grad_params = self.treelstm:getParameters()
end

function MergeTreeLSTMSim:new_sim_module()

  --mlp = HighwayMLP.mlp(self.mem_dim, 2)
   -- define similarity model architecture
  local sim_module = nn.Sequential()
    --:add(nn.Linear(self.mem_dim, self.sim_nhidden))
    --:add(nn.Sigmoid())    -- does better than tanh
    --:add(nn.Linear(self.sim_nhidden, self.num_classes))
    --:add(mlp)
    :add(nn.Linear(self.mem_dim, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module
end

function MergeTreeLSTMSim:train(dataset)
  self.treelstm:training()
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
        local mtree = dataset.mtrees[idx]
        local msent = dataset.msents[idx]


        local minputs = self.emb:forward(msent)

        -- get sentence representations    
        local _,tree_loss = self.treelstm:forward(mtree, minputs)
        loss = loss + tree_loss

        local minput_grads = self.treelstm:backward(mtree, minputs, {zeros, zeros})
        self.emb:backward(msent, minput_grads)
      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)
      self.emb.gradWeight:div(batch_size)
      self.emb:updateParameters(self.emb_learning_rate)

      -- regularization
      loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end

    optim.adagrad(feval, self.params, self.optim_state)
  end
  xlua.progress(dataset.size, dataset.size)
end

function MergeTreeLSTMSim:predict(tree, sent)
  self.treelstm:evaluate()
  local prediction
  local inputs = self.emb:forward(sent)
  local mrep=self.treelstm:forward(tree, inputs)[2]
  --local output = self.sim_module:forward(mrep)
  prediction = argmax(tree.output)
  self.treelstm:clean(tree)
  return prediction
end

function MergeTreeLSTMSim:predict_dataset(dataset)
  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    predictions[i] = self:predict(dataset.mtrees[i], dataset.msents[i])
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

function MergeTreeLSTMSim:print_config()
  local num_params = self.params:size(1)
  local num_sim_params = self:new_sim_module():getParameters():size(1)
  printf('%-25s = %d\n',   'num params', num_params)
  printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'MergeTree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end