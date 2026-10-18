import os
import torch

ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
best_path = os.path.join(ckpt_dir, 'best_model.pth')
final_path = os.path.join(ckpt_dir, 'final_model.pth')

best_ckpt = torch.load(best_path, map_location='cpu')
print('=== BEST MODEL METADATA ===')
if isinstance(best_ckpt, dict):
    for k in best_ckpt.keys():
        if k != 'optimizer_state_dict' and not k.endswith('state_dict'):
            print(f'  {k}: {best_ckpt[k]}')
    print('  Keys:', list(best_ckpt.keys()))

final_ckpt = torch.load(final_path, map_location='cpu')
print('\n=== FINAL MODEL METADATA ===')
if isinstance(final_ckpt, dict):
    for k in final_ckpt.keys():
        if k != 'optimizer_state_dict' and not k.endswith('state_dict'):
            print(f'  {k}: {final_ckpt[k]}')
    print('  Keys:', list(final_ckpt.keys()))

def get_sd(ckpt):
    for k in ['policy_state_dict', 'policy_net_state_dict', 'model_state_dict']:
        if k in ckpt:
            return ckpt[k]
    return ckpt

best_sd = get_sd(best_ckpt)
final_sd = get_sd(final_ckpt)

print('\n=== WEIGHT NORMS COMPARISON ===')
for key in best_sd.keys():
    if 'weight' in key and key in final_sd:
        b_norm = best_sd[key].float().norm().item()
        f_norm = final_sd[key].float().norm().item()
        diff = (final_sd[key].float() - best_sd[key].float()).norm().item()
        print(f'{key:35s} | Best: {b_norm:7.3f} | Final: {f_norm:7.3f} | Diff: {diff:7.3f}')
