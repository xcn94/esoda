import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
bert_path = '/root/xcn/bert_pytorch'
tokenizer = BertTokenizer.from_pretrained(bert_path)
text = 'However, it _ too late'
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)

masked_index = tokenized_text.index('_')
tokenized_text[masked_index] = '[MASK]'

candidates = ['love', 'work', 'enjoy', 'play']
candidates_ids = tokenizer.convert_tokens_to_ids(candidates)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

segments_ids = [0] * len(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

language_model = BertForMaskedLM.from_pretrained(bert_path)
# language_model.eval()

predictions = language_model(tokens_tensor, segments_tensors)
predictions_candidates = predictions[0, masked_index]
print(predictions_candidates)
answer_idx = torch.argmax(predictions_candidates).item()
print(answer_idx)
print(f'The most likely word is "{tokenizer.convert_ids_to_tokens([answer_idx])[0]}".')
