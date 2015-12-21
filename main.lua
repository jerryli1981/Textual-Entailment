require 'init' 

function accuracy(pred,gold)
  return torch.eq(pred, gold):sum()/pred:size(1)
end

local args = lapp [[
Training script for semantic relatedness prediction on the SICK dataset.
  -m,--model  (default treeLSTM)  Model architecture: [treeLSTM, seqLSTM]
  -d,--dim    (default 10)        LSTM memory dimension
  -e,--epochs (default 10)        Number of training epochs
  -l,--num_layers (default 1)          Number of layers
  -h,--sim_nhidden (default 50)    number of sim_hidden
  -s,--structure (default lstm)   lstm structure
  -g,--debug  (default nil)       debug setting   

]]

if args.debug == 'dbg' then
	dbg = require('debugger')
end

local data_dir = 'data/sick/'
local vocab = Vocab(data_dir .. 'vocab-cased.txt')

local emb_dir = 'data/glove/'
local emb_prefix = emb_dir .. 'glove.840B'
local emb_vocab, emb_vecs = read_embedding(emb_prefix .. '.vocab', emb_prefix .. '.300d.th')
local emb_dim = emb_vecs:size(2)

local num_unk=0

local vecs = torch.Tensor(vocab.size, emb_dim)

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
  structure = args.structure,
  num_layers = args.num_layers,
  sim_nhidden = args.sim_nhidden
}

-- number of epochs to train
local num_epochs = args.epochs

-- print information
header('model configuration')
printf('max epochs = %d\n', num_epochs)
model:print_config()

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
  local dev_score = accuracy(dev_predictions, dev_dataset.labels)

  printf('--dev_accuracy: %.4f\n', dev_score)

  if dev_score >= best_dev_score then
    best_dev_score = dev_score

    best_dev_model = model_class{
      emb_vecs   = vecs,
      mem_dim    = args.dim,
      structure = args.structure,
      num_layers = args.num_layers,
      sim_nhidden = args.sim_nhidden
    }

    best_dev_model.params:copy(model.params)

  end
end

printf('-- best dev score: %.4f\n', best_dev_score)

printf('finished training in %.2fs\n', sys.clock() - train_start)

-- evaluate
header('Evaluating on test set')
printf('-- using model with dev score = %.4f\n', best_dev_score)
local test_predictions = best_dev_model:predict_dataset(test_dataset)
local test_score = accuracy(test_predictions, test_dataset.labels)
printf('-- test score: %.4f\n', test_score)


-- create predictions and model directories if necessary
if lfs.attributes(predictions_dir) == nil then
  lfs.mkdir(predictions_dir)
end

if lfs.attributes(models_dir) == nil then
  lfs.mkdir(models_dir)
end

-- get paths
local file_idx = 1
local predictions_save_path, model_save_path
while true do
  predictions_save_path = string.format(
    predictions_dir .. '/ent-%s.%dl.%dd.%d.pred', args.model, args.num_layers, args.dim, file_idx)
  model_save_path = string.format(
    models_dir .. '/ent-%s.%dl.%dd.%d.th', args.model, args.num_layers, args.dim, file_idx)
  if lfs.attributes(predictions_save_path) == nil and lfs.attributes(model_save_path) == nil then
    break
  end
  file_idx = file_idx + 1
end

-- write predictions to disk
local predictions_file = torch.DiskFile(predictions_save_path, 'w')
print('writing predictions to ' .. predictions_save_path)
for i = 1, test_predictions:size(1) do
  predictions_file:writeFloat(test_predictions[i])
end
predictions_file:close()


--write models to disk
print('writing model to ' .. model_save_path)
best_dev_model:save(model_save_path)

--to load a saved model
--local loaded = model_class.load(model_save_path)
