function read_embedding(vocab_path, emb_path)
  local vocab = Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end

function read_trees(parent_path, labels)
  local parent_file = io.open(parent_path, 'r')

  local count = 0
  local trees = {}
  
  while true do
    local parents = parent_file:read()
    if parents == nil then break end
    parents = stringx.split(parents)
    for i, p in ipairs(parents) do
      parents[i] = tonumber(p)
    end

    count = count + 1

    trees[count] = read_tree(parents, labels[count])
  end
  parent_file:close()
  return trees
end

function read_tree(parents, label)
  local size = #parents
  local trees = {}
  local root
  for i = 1, size do
    if not trees[i] and parents[i] ~= -1 then
      local idx = i
      local prev = nil
      while true do
        local parent = parents[idx]
        if parent == -1 then
          break
        end

        local tree = Tree()
        if prev ~= nil then
          tree:add_child(prev)
        end
        trees[idx] = tree
        tree.idx = idx
        tree.gold_label = label
        if trees[parent] ~= nil then
          trees[parent]:add_child(tree)
          break
        elseif parent == 0 then
          root = tree
          break
        else
          prev = tree
          idx = parent
        end
      end
    end
  end

  return root
end

function read_dataset(dir, vocab)
  local labelMap = {NEUTRAL=3, CONTRADICTION=1, ENTAILMENT=2}
  local dataset = {}
  dataset.vocab = vocab
  
  --dataset.msents = read_sentences(dir .. 'm.toks', vocab)
  dataset.lsents = read_sentences(dir .. 'a.toks', vocab)
  dataset.rsents = read_sentences(dir .. 'b.toks', vocab)
  dataset.size = #dataset.lsents
  local id_file = torch.DiskFile(dir .. 'id.txt')
  local label_file = io.open(dir .. 'label.txt', 'r')
  dataset.ids = torch.IntTensor(dataset.size)
  dataset.labels = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    dataset.ids[i] = id_file:readInt()
    dataset.labels[i] = labelMap[label_file:read()]
  end

  dataset.ltrees = read_trees(dir .. 'a.parents', dataset.labels)
  dataset.rtrees = read_trees(dir .. 'b.parents', dataset.labels)
  --dataset.mtrees = read_trees(dir .. 'm.parents', dataset.labels)

  id_file:close()
  label_file:close()
  return dataset
end