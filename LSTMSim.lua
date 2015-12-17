local LSTMSim = torch.class('LSTMSim')

function LSTMSim:__init(config)
  self.mem_dim       = config.mem_dim       or 150
  self.learning_rate = config.learning_rate or 0.05
  self.batch_size    = config.batch_size    or 25
  self.reg           = config.reg           or 1e-4
  self.sim_nhidden   = config.sim_nhidden   or 50
  self.structure     = config.structure     or 'lstm'
  self.num_layers    = config.num_layers    or 1

  -- word embedding
  self.emb_dim = config.emb_vecs:size(2)
  self.emb_vecs = config.emb_vecs

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
    self.llstm = LSTM(lstm_config) -- "left" LSTM
    self.rlstm = LSTM(lstm_config) -- "right" LSTM
  elseif self.structure == 'bilstm' then
    self.llstm = LSTM(lstm_config)
    self.llstm_b = LSTM(lstm_config) -- backward "left" LSTM
    self.rlstm = LSTM(lstm_config)
    self.rlstm_b = LSTM(lstm_config) -- backward "right" LSTM
  else
    error('invalid LSTM type: ' .. self.structure)
  end

  self.sim_module = self:new_sim_module_conv1d()

  local modules = nn.Parallel()
    :add(self.llstm)
    :add(self.sim_module)

  self.params, self.grad_params = modules:getParameters()

  -- share must only be called after getParameters, since this changes the
  -- location of the parameters
  share_params(self.rlstm, self.llstm)
  if self.structure == 'bilstm' then
    -- tying the forward and backward weights improves performance
    share_params(self.llstm_b, self.llstm)
    share_params(self.rlstm_b, self.llstm)
  end
end

function LSTMSim:new_sim_module()
  print('Using simple sim module')
  local lvec, rvec, inputs, input_dim
  if self.structure == 'lstm' then
    -- standard (left-to-right) LSTM
    input_dim = 2 * self.num_layers * self.mem_dim
    local linput, rinput = nn.Identity()(), nn.Identity()()
    if self.num_layers == 1 then
      lvec, rvec = linput, rinput
    else
      lvec, rvec = nn.JoinTable(1)(linput), nn.JoinTable(1)(rinput)
    end
    inputs = {linput, rinput}
  elseif self.structure == 'bilstm' then
    -- bidirectional LSTM
    input_dim = 4 * self.num_layers * self.mem_dim
    local lf, lb, rf, rb = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()
    if self.num_layers == 1 then
      lvec = nn.JoinTable(1){lf, lb}
      rvec = nn.JoinTable(1){rf, rb}
    else
      -- in the multilayer case, each input is a table of hidden vectors (one for each layer)
      lvec = nn.JoinTable(1){nn.JoinTable(1)(lf), nn.JoinTable(1)(lb)}
      rvec = nn.JoinTable(1){nn.JoinTable(1)(rf), nn.JoinTable(1)(rb)}
    end
    inputs = {lf, lb, rf, rb}
  end

  local mult_dist = nn.CMulTable(){lvec, rvec}
  local add_dist = nn.Abs()(nn.CSubTable(){lvec, rvec})
  local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
  local vecs_to_input = nn.gModule(inputs, {vec_dist_feats})

   -- define similarity model architecture
  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(input_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())
  return sim_module
end

function LSTMSim:new_sim_module_conv2d()
  print('Using conv2d sim module, num_layers must > 2')
  local lmat, rmat, inputs
  
  local num_plate

  if self.structure == 'lstm' then

    local linput, rinput = nn.Identity()(), nn.Identity()()
    
    local linput_r,rinput_r = {},{}
    for i=1, self.num_layers do

      l_i_vec = nn.SelectTable(i)(linput)
      l_i_vec = nn.Reshape(1, self.mem_dim)(l_i_vec)
      linput_r[i] = l_i_vec

      r_i_vec = nn.SelectTable(i)(rinput)
      r_i_vec = nn.Reshape(1, self.mem_dim)(r_i_vec)
      rinput_r[i] = r_i_vec
    end

    lmat, rmat = nn.JoinTable(1)(linput_r), nn.JoinTable(1)(rinput_r)
    inputs = {linput, rinput}

  elseif self.structure == 'bilstm' then

    local lf, lb, rf, rb = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()

    --[[
    local lf_r, lb_r, rf_r, rb_r = {},{},{},{}
    for i=1, self.num_layers do

      lf_i_vec = nn.SelectTable(i)(lf)
      lf_i_vec = nn.Reshape(1, self.mem_dim)(lf_i_vec)
      lf_r[i] = lf_i_vec

      lb_i_vec = nn.SelectTable(i)(lb)
      lb_i_vec = nn.Reshape(1, self.mem_dim)(lb_i_vec)
      lb_r[i] = lb_i_vec

      rf_i_vec = nn.SelectTable(i)(rf)
      rf_i_vec = nn.Reshape(1, self.mem_dim)(rf_i_vec)
      rf_r[i] = rf_i_vec

      rb_i_vec = nn.SelectTable(i)(rb)
      rb_i_vec = nn.Reshape(1, self.mem_dim)(rb_i_vec)
      rb_r[i] = rb_i_vec
    end
    --]]

    lmat = nn.JoinTable(1){nn.JoinTable(1)(lf), nn.JoinTable(1)(lb)}
    rmat = nn.JoinTable(1){nn.JoinTable(1)(rf), nn.JoinTable(1)(rb)}

    --lf_mat = nn.JoinTable(1)(lf_r)
    --lb_mat = nn.JoinTable(1)(lb_r)
    --lmat = nn.JoinTable(1){lf_mat, lb_mat}

    --rf_mat = nn.JoinTable(1)(rf_r)
    --rb_mat = nn.JoinTable(1)(rb_r)
    --rmat = nn.JoinTable(1){rf_mat, rb_mat}

    inputs = {lf, lb, rf, rb}
  end

  local img_h = self.num_layers
  local img_w = self.mem_dim

  local mult_dist = nn.CMulTable(){lmat, rmat}

  local add_dist = nn.Abs()(nn.CSubTable(){lmat, rmat})

  --local radial_dist = nn.Exp()(nn.MulConstant(-0.25)(nn.Power(2)(add_dist)))

  local max_dist = nn.Max(1)(nn.Reshape(2,self.mem_dim*2*img_h)(nn.JoinTable(1){lmat, rmat}))

  local out_mat = nn.JoinTable(1){mult_dist, add_dist, max_dist}

  num_plate = 6

  out_mat = nn.Reshape(num_plate, img_h, img_w){out_mat}

  vecs_to_input = nn.gModule(inputs, {out_mat})

  local conv_kw = img_w
  local conv_kh = 2
  local n_input_plane = num_plate
  local n_output_plane = 2
  local pool_kw = 1
  local pool_kh = 2

  local cov_out_h = img_h - conv_kh + 1
  local cov_out_w = img_w - conv_kw + 1
  local pool_out_h = cov_out_h - pool_kh + 1
  local pool_out_w = cov_out_w -pool_kw + 1

  local mlp_input_dim = n_output_plane*pool_out_h*pool_out_w

  --local mlp_input_dim = 4 * self.num_layers * self.mem_dim

  local sim_module = nn.Sequential()
    :add(vecs_to_input)
    --:add(nn.SpatialConvolution(n_input_plane, n_output_plane, conv_kw, conv_kh))
    
    :add(nn.LateralConvolution(n_input_plane, n_output_plane))
    :add(nn.VerticalConvolution(n_output_plane, n_output_plane, conv_kh))
    :add(nn.HorizontalConvolution(n_output_plane, n_output_plane, conv_kw))
    :add(nn.Tanh())
    :add(nn.SpatialMaxPooling(pool_kw, pool_kh, 1, 1))
    :add(nn.SpatialSubtractiveNormalization(n_output_plane, image.gaussian1D(7)))
    :add(nn.Reshape(mlp_input_dim))
    :add(HighwayMLP.mlp(mlp_input_dim, 1))
    :add(nn.Linear(mlp_input_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())

  return sim_module

end

function LSTMSim:new_sim_module_conv1d()
  print('Using conv1d sim module, num_layers must > 2')
  local lmat, rmat, inputs
  
  local num_plate

  if self.structure == 'lstm' then

    local linput, rinput = nn.Identity()(), nn.Identity()()
    
    local linput_r,rinput_r = {},{}
    for i=1, self.num_layers do

      l_i_vec = nn.SelectTable(i)(linput)
      l_i_vec = nn.Reshape(1, self.mem_dim)(l_i_vec)
      linput_r[i] = l_i_vec

      r_i_vec = nn.SelectTable(i)(rinput)
      r_i_vec = nn.Reshape(1, self.mem_dim)(r_i_vec)
      rinput_r[i] = r_i_vec
    end

    lmat, rmat = nn.JoinTable(1)(linput_r), nn.JoinTable(1)(rinput_r)
    inputs = {linput, rinput}

  elseif self.structure == 'bilstm' then

    local lf, lb, rf, rb = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()

    lmat = nn.JoinTable(1){nn.JoinTable(1)(lf), nn.JoinTable(1)(lb)}
    rmat = nn.JoinTable(1){nn.JoinTable(1)(rf), nn.JoinTable(1)(rb)}

    inputs = {lf, lb, rf, rb}
  end

  local img_h = self.num_layers
  local img_w = self.mem_dim

  local mult_dist = nn.CMulTable(){lmat, rmat}
  local add_dist = nn.Abs()(nn.CSubTable(){lmat, rmat})
  local max_dist = nn.Max(1)(nn.Reshape(2,self.mem_dim*2*img_h)(nn.JoinTable(1){lmat, rmat}))

  local out_mat = nn.JoinTable(1){mult_dist, add_dist, max_dist}

  num_plate = 6

  out_mat = nn.Reshape(num_plate, img_h*img_w){out_mat}

  vecs_to_input = nn.gModule(inputs, {out_mat})

  local inputFrameSize = img_h*img_w
  local outputFrameSize = 50
  local kw = 3

  local pool_kw = 2

  local mlp_input_dim = (num_plate-kw+1-pool_kw+1) * outputFrameSize
  --local outputFrameSize2 = 10
  --local kw2=2
  --local mlp_input_dim2 = (num_plate-kw+1-pool_kw+1-kw2+1-pool_kw+1) * 10

  local sim_module = nn.Sequential()
    :add(vecs_to_input) -- 6 * 200
    :add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kw))
    :add(nn.Tanh())
    :add(nn.TemporalMaxPooling(pool_kw, 1))

    --:add(nn.TemporalConvolution(outputFrameSize, outputFrameSize2, kw2))
    --:add(nn.Tanh())
    --:add(nn.TemporalMaxPooling(pool_kw, 1))
    --:add(nn.Reshape(mlp_input_dim2))
    --:add(HighwayMLP.mlp(mlp_input_dim2, 1))
    --:add(nn.Linear(mlp_input_dim2, self.sim_nhidden))

    :add(nn.Reshape(mlp_input_dim))
    :add(HighwayMLP.mlp(mlp_input_dim, 1))
    :add(nn.Linear(mlp_input_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, self.num_classes))
    :add(nn.LogSoftMax())

  return sim_module

end

function LSTMSim:train(dataset)

  self.llstm:training()
  self.rlstm:training()

  if self.structure == 'bilstm' then
    self.llstm_b:training()
    self.rlstm_b:training()
  end

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

        local linputs = self.emb_vecs:index(1, lsent:long()):double()
        local rinputs = self.emb_vecs:index(1, rsent:long()):double()

         -- get sentence representations
        local inputs
        if self.structure == 'lstm' then
          inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
        elseif self.structure == 'bilstm' then
          inputs = {
            self.llstm:forward(linputs),
            self.llstm_b:forward(linputs, true), -- true => reverse
            self.rlstm:forward(rinputs),
            self.rlstm_b:forward(rinputs, true)
          }
        end

        local output = self.sim_module:forward(inputs)
        --dbg()
        
  
        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, ent)

        loss = loss + example_loss

        local sim_grad = self.criterion:backward(output, ent)
        local rep_grad = self.sim_module:backward(inputs, sim_grad)

        if self.structure == 'lstm' then
          self:LSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
        elseif self.structure == 'bilstm' then
          self:BiLSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
        end

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

-- LSTM backward propagation
function LSTMSim:LSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
  local lgrad, rgrad
  if self.num_layers == 1 then
    lgrad = torch.zeros(lsent:nElement(), self.mem_dim)
    rgrad = torch.zeros(rsent:nElement(), self.mem_dim)
    lgrad[lsent:nElement()] = rep_grad[1]
    rgrad[rsent:nElement()] = rep_grad[2]
  else
    lgrad = torch.zeros(lsent:nElement(), self.num_layers, self.mem_dim)
    rgrad = torch.zeros(rsent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      lgrad[{lsent:nElement(), l, {}}] = rep_grad[1][l]
      rgrad[{rsent:nElement(), l, {}}] = rep_grad[2][l]
    end
  end
  self.llstm:backward(linputs, lgrad)
  self.rlstm:backward(rinputs, rgrad)
end

-- Bidirectional LSTM backward propagation
function LSTMSim:BiLSTM_backward(lsent, rsent, linputs, rinputs, rep_grad)
  local lgrad, lgrad_b, rgrad, rgrad_b
  if self.num_layers == 1 then
    lgrad   = torch.zeros(lsent:nElement(), self.mem_dim)
    lgrad_b = torch.zeros(lsent:nElement(), self.mem_dim)
    rgrad   = torch.zeros(rsent:nElement(), self.mem_dim)
    rgrad_b = torch.zeros(rsent:nElement(), self.mem_dim)
    lgrad[lsent:nElement()] = rep_grad[1]
    rgrad[rsent:nElement()] = rep_grad[3]
    lgrad_b[1] = rep_grad[2]
    rgrad_b[1] = rep_grad[4]
  else
    lgrad   = torch.zeros(lsent:nElement(), self.num_layers, self.mem_dim)
    lgrad_b = torch.zeros(lsent:nElement(), self.num_layers, self.mem_dim)
    rgrad   = torch.zeros(rsent:nElement(), self.num_layers, self.mem_dim)
    rgrad_b = torch.zeros(rsent:nElement(), self.num_layers, self.mem_dim)
    for l = 1, self.num_layers do
      lgrad[{lsent:nElement(), l, {}}] = rep_grad[1][l]
      rgrad[{rsent:nElement(), l, {}}] = rep_grad[3][l]
      lgrad_b[{1, l, {}}] = rep_grad[2][l]
      rgrad_b[{1, l, {}}] = rep_grad[4][l]
    end
  end
  self.llstm:backward(linputs, lgrad)
  self.llstm_b:backward(linputs, lgrad_b, true)
  self.rlstm:backward(rinputs, rgrad)
  self.rlstm_b:backward(rinputs, rgrad_b, true)
end

-- Predict the similarity of a sentence pair.
function LSTMSim:predict(lsent, rsent)
  self.llstm:evaluate()
  self.rlstm:evaluate()
  local linputs = self.emb_vecs:index(1, lsent:long()):double()
  local rinputs = self.emb_vecs:index(1, rsent:long()):double()
  local inputs
  if self.structure == 'lstm' then
    inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
  elseif self.structure == 'bilstm' then
    self.llstm_b:evaluate()
    self.rlstm_b:evaluate()
    inputs = {
      self.llstm:forward(linputs),
      self.llstm_b:forward(linputs, true),
      self.rlstm:forward(rinputs),
      self.rlstm_b:forward(rinputs, true)
    }
  end
  local output = self.sim_module:forward(inputs)
  local prediction = argmax(output)
  self.llstm:forget()
  self.rlstm:forget()
  if self.structure == 'bilstm' then
    self.llstm_b:forget()
    self.rlstm_b:forget()
  end
  return prediction
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

-- Produce similarity predictions for each sentence pair in the dataset.
function LSTMSim:predict_dataset(dataset)

  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(lsent, rsent)
  end
  return predictions
end

function LSTMSim:print_config()
  printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
  printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
  printf('%-25s = %.2e\n', 'regularization strength', self.reg)
  printf('%-25s = %d\n',   'minibatch size', self.batch_size)
  printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
  printf('%-25s = %s\n',   'LSTM structure', self.structure)
  printf('%-25s = %d\n',   'LSTM layers', self.num_layers)
  printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
end