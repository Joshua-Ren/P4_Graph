import torch

# ========== Message and its evaluations ===============
def cal_msg_distance_model(args, student, teacher, batch):
  with torch.no_grad():
    student.eval()
    teacher.eval()
    stud_logits = student.distill_forward(batch)
    teach_logits = teacher.distill_forward(batch)
    dist = (stud_logits.argmax(-1)==teach_logits.argmax(-1)).sum()/(stud_logits.shape[0]*stud_logits.shape[1])
    return dist

def cal_msg_distance_logits(logits1, logits2):
  with torch.no_grad():
    dist = (logits1.argmax(-1)==logits2.argmax(-1)).sum()/(logits1.shape[0]*logits1.shape[1])
    return dist

# ============= Graph data augmentation =============
def get_batch_aug(args, batch, aug_type='node_gaussian'):
  # Give one batch data, generate two augmentation versions on X
  # For graph, that can be a little complex
  batch1 = copy.deepcopy(batch)
  batch2 = copy.deepcopy(batch)
  if aug_type=='node_gaussian':
    noisy_on1 = torch.tensor(np.random.uniform(low=0.9, high=1.0, size=batch.x.size()))
    noisy_on2 = torch.tensor(np.random.uniform(low=0.9, high=1.0, size=batch.x.size()))
    batch1.x = (batch.x*noisy_on1).int()
    batch2.x = (batch.x*noisy_on2).int()
  elif aug_type=='edge_drop':
    pass
  return batch1, batch2

# ============= Model parameters update =============
class EMA():
  def __init__(self, eta):
    super().__init__()
    self.eta = eta

  def update_average(self, old, new):
    if old is None:
      return new
    return old * self.eta + (1 - self.eta) * new
    
def EMA_update(online_model, target_model, eta=0.99):
  ema = EMA(eta)
  with torch.no_grad():
    for online_params, target_params in zip(online_model.parameters(),target_model.parameters()):
      old_weight, up_weight = target_params.data, online_params.data
      target_params.data = ema.update_average(old_weight, up_weight)