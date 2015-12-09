local TreeLSTMSim = torch.class('TreeLSTMSim')

function TreeLSTMSim:__init(config)
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
    gate_output = false,
  }
  

  self.treelstm = ChildSumTreeLSTM(treelstm_config)

  -- similarity model
  self.sim_module = self:new_sim_module()

  local modules = nn.Parallel()
    :add(self.treelstm)
    :add(self.sim_module)
  self.params, self.grad_params = modules:getParameters()
end

function TreeLSTMSim:new_sim_module()
  local vecs_to_input
  local lvec = nn.Identity()()
  local rvec = nn.Identity()()
  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local cosine_dist = nn.CosineDistance(){lvec, rvec}
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  local vec_dist_feats_1 = nn.JoinTable(1){vec_dist_feats, cosine_dist}
  vecs_to_input = nn.gModule({lvec, rvec}, {vec_dist_feats_1})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(2 * self.mem_dim+1, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module
end


function TreeLSTMSim:new_sim_module_CNN()
  local seq_length = 72
  local inputFrameSize = 10
  local outputFrameSize = 10
  local kernel_width = 3
  local reduced_l = seq_length - kernel_width + 1 

  local vecs_to_input
  local lmat = nn.Identity()()
  local rmat = nn.Identity()()
  local vec_dist_feats = nn.JoinTable(1){lmat, rmat}
  vecs_to_input = nn.gModule({lmat, rmat}, {vec_dist_feats})

  --[[
  sim_matrix = torch.randn(36,36)

        for i =1, 36 do
            vec_l = matrix_l[i]
          for j=1, 36 do
            vec_r = matrix_r[j]
            sim_val = nn.CosineDistance():forward{vec_l, vec_r}
            sim_matrix[i][j] = sim_val
          end
        end
  --]]

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kernel_width))--34 * 10
    :add(nn.Tanh())
    :add(nn.TemporalMaxPooling(reduced_l)) --1*10
    :add(nn.Linear(outputFrameSize, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module

end

function TreeLSTMSim:train(dataset)
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
        local ltree, rtree = dataset.ltrees[idx], dataset.rtrees[idx]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]
        local ent = dataset.labels[idx]

        self.emb:forward(lsent)
        local linputs = torch.Tensor(self.emb.output:size()):copy(self.emb.output)
        local rinputs = self.emb:forward(rsent)
        
        local lrep = self.treelstm:forward(ltree, linputs)[2]
        local rrep = self.treelstm:forward(rtree, rinputs)[2]
        local output = self.sim_module:forward{lrep, rrep}

        local example_loss = self.criterion:forward(output, ent)
        loss = loss + example_loss
        local sim_grad = self.criterion:backward(output, ent)
        local rep_grad = self.sim_module:backward({lrep, rrep}, sim_grad)

        local linput_grads = self.treelstm:backward(dataset.ltrees[idx], linputs, {zeros, rep_grad[1]})
        local rinput_grads = self.treelstm:backward(dataset.rtrees[idx], rinputs, {zeros, rep_grad[2]})

        self.emb:backward(lsent, linput_grads)
        self.emb:backward(rsent, rinput_grads)
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

-- Predict the similarity of a sentence pair.
function TreeLSTMSim:predict(ltree, rtree, lsent, rsent)
  local linputs = self.emb:forward(lsent)
  local lrep = self.treelstm:forward(ltree, linputs)[2]
  local rinputs = self.emb:forward(rsent)
  local rrep = self.treelstm:forward(rtree, rinputs)[2]
  local output = self.sim_module:forward{lrep, rrep}

  local prediction = argmax(output)
  self.treelstm:clean(ltree)
  self.treelstm:clean(rtree)
  return prediction
end

-- Produce similarity predictions for each sentence pair in the dataset.
function TreeLSTMSim:predict_dataset(dataset)
  self.treelstm:evaluate()

  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local ltree, rtree = dataset.ltrees[i], dataset.rtrees[i]
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(ltree, rtree, lsent, rsent)
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

function TreeLSTMSim:print_config()
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'Tree-LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %.2e\n', 'word vector learning rate', self.emb_learning_rate)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end