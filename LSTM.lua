
-- single layer lstm
local LSTM, parent = torch.class('LSTM', 'nn.Module')

function LSTM:__init(config)
	parent.__init(self)

	self.in_dim = config.in_dim
	self.mem_dim = config.mem_dim or 150


	self.master_cell = self:new_cell()

	self.cells = {}

	self.depth = 0

	local ctable_init, ctable_grad, htable_init, htable_grad

	ctable_init = torch.zeros(self.mem_dim)
	ctable_grad = torch.zeros(self.mem_dim)
	htable_init = torch.zeros(self.mem_dim)
	htable_grad = torch.zeros(self.mem_dim)

	self.initial_values = {ctable_init, htable_init}

	self.gradInput = {
		torch.zeros(self.in_dim),
		ctable_grad,
		htable_grad
	}
end

function LSTM:new_cell() 
	local input = nn.Identity()()
	local cell_prev = nn.Identity()()
	local hidden_prev = nn.Identity()()

	local new_gate = function()
		local in_module = nn.Linear(self.in_dim, self.mem_dim)(input)
		return nn.CAddTable(){in_module, nn.Linear(self.mem_dim, self.mem_dim)(hidden_prev)}
	end

	local i_gate = nn.Sigmoid()(new_gate())
	local f_gate = nn.Sigmoid()(new_gate())
	local update_gate = nn.Tanh()(new_gate())
	local o_gate = nn.Sigmoid()(new_gate())

	cell = nn.CAddTable(){
		nn.CMulTable(){f_gate, cell_prev},
		nn.CMulTable(){i_gate, update_gate}
	}

	hidden = nn.CMulTable(){o_gate, nn.Tanh()(cell)}

	local mod = nn.gModule({input, cell_prev, hidden_prev}, {cell, hidden})

	if self.master_cell then
		share_params(mod, self.master_cell)
	end

	return mod

end

--inputs: T*in_dim tensor, where T is the number of time steps
--reverse: if true, read the input from right to left
--Returns the final hidden state of the LSTM ?
function LSTM:forward(inputs, reverse)
	local size = inputs:size(1)
	self.output = torch.Tensor(36, self.mem_dim):zero()
	for t = 1, size do
		local input = reverse and inputs[size-t+1] or inputs[t]
		self.depth = self.depth +1
		local cell = self.cells[self.depth]
		if cell == nil then
			cell = self:new_cell()
			self.cells[self.depth] = cell
		end

		local prev_output

		if self.depth > 1 then
			prev_output = self.cells[self.depth -1].output
		else
			prev_output = self.initial_values
		end 

		local outputs = cell:forward({input, prev_output[1], prev_output[2]})
		local cell, hidden = unpack(outputs)
		self.output[t + 36 -size] = hidden
		--self.output=hidden
	end

	return self.output
end

--grad_outputs: T * mem_dim tersor
function LSTM:backward(inputs, grad_outputs, reverse)
	local size = inputs:size(1)
	if self.depth == 0 then
		error("No cells to backpropagate through")
	end

	local input_grads = torch.Tensor(inputs:size())

	for t = size, 1, -1 do
		local input = reverse and inputs[size -t + 1] or inputs[t]
		local grad_output = reverse and grad_outputs[size -t + 1] or grad_outputs[t]
		--local input = reverse and inputs[size -t + 1] or inputs[t+ 36 -size]
		--local grad_output = reverse and grad_outputs[size -t + 1] or grad_outputs[t+ 36 -size]
		local cell = self.cells[self.depth]
		local grads = {self.gradInput[2], grad_output}

		local prev_output = (self.depth > 1) and self.cells[self.depth -1].output
											 or self.initial_values

		self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)

		if reverse then
			input_grads[size-t +1] = self.gradInput[1]
		else
			input_grads[t] = self.gradInput[1]
		end
		self.depth = self.depth-1
	end
	self:forget()
	return input_gards
end

function LSTM:share(lstm, ...)
  if self.in_dim ~= lstm.in_dim then error("LSTM input dimension mismatch") end
  if self.mem_dim ~= lstm.mem_dim then error("LSTM memory dimension mismatch") end
  if self.gate_output ~= lstm.gate_output then error("LSTM output gating mismatch") end
  share_params(self.master_cell, lstm.master_cell, ...)
end

function LSTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function LSTM:parameters()
  return self.master_cell:parameters()
end

function LSTM:forget()
	self.depth = 0
	for i = 1, #self.gradInput do
		local gradInput = self.gradInput[i]
		if type(gradInput) == 'table' then
			for _, t in pairs(gradInput) do t:zero() end
		else
			self.gradInput[i]:zero()
		end
	end
end















