require 'init' 


local args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -m,--model  (default treeLSTM) Model architecture: [treeLSTM, seqLSTM]
  -d,--dim    (default 10)        LSTM memory dimension
  -e,--epochs (default 10)         Number of training epochs
  -g,--debug  (default nil)        debug setting   
  -c,--cuda   (default nil)       cuda setting
]]

if args.debug == 'dbg' then
	dbg = require('debugger')
end


localize = function(thing)
  if args.cuda == 'gpu' then
    require('cutorch')
    return thing:cuda()
  end
  return thing
end


local data_dir = 'data/sick/'
local vocab = Vocab(data_dir .. 'vocab-cased.txt')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

local num_unk=0

local vecs = localize(torch.Tensor(vocab.size, emb_dim))
--dbg()
for i = 1, vocab.size do
	local w = vocab:token(i)
	if emb_vocab:contains(w) then
		vecs[i] = emb_vecs[emb_vocab:index(w)]
	else
		num_unk = num_unk +1
		vecs[i]:uniform(-0.05, 0.05)
	end
end
print('unk count = ' ..num_unk)
emb_vocab = nil
emb_vecs = nil
collectgarbage()

print('loading datasets')
local train_dir = data_dir .. 'train/'
local dev_dir = data_dir .. 'dev/'
local test_dir = data_dir .. 'test/'
local train_dataset = read_dataset(train_dir, vocab)
local dev_dataset = read_dataset(dev_dir, vocab)
local test_dataset = read_dataset(test_dir, vocab)

printf('num train = %d\n', train_dataset.size)
printf('num dev = %d\n', dev_dataset.size)
printf('num test = %d\n', test_dataset.size)


--A one-table-param function call needs no parens:
-- initialize model
if args.model == "treeLSTM" then
  model_class = TreeLSTMSim
elseif args.model == "seqLSTM" then
  model_class = LSTMSim
end

local model = model_class{
  emb_vecs   = vecs,
  mem_dim    = args.dim,
}

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

classes = {'1','2','3'}

confusion = optim.ConfusionMatrix(classes)

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
header('Training model')
for i = 1, num_epochs do
  local start = sys.clock()
  printf('-- epoch %d\n', i)
  model:train(train_dataset)
  printf('-- finished epoch in %.2fs\n', sys.clock() - start)

  local dev_predictions = model:predict_dataset(dev_dataset)

  confusion:batchAdd(dev_predictions, dev_dataset.labels)
  confusion:updateValids()
  dev_score = confusion.totalValid * 100
  printf('--dev_accuracy: %.4f\n', dev_score)
end

printf('finished training in %.2fs\n', sys.clock() - train_start)