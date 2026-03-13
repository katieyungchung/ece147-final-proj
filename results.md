## Best Result w/ Beam Decoder

| Validate metric | DataLoader 0          |
|-----------------|----------------------|
| val/CER         | 11.763402938842773   |
| val/DER         | 1.7058041095733643   |
| val/IER         | 2.835622549057007    |
| val/SER         | 7.221976280212402    |
| val/loss        | 0.5399672389030457   |

| Test metric     | DataLoader 0          |
|-----------------|----------------------|
| test/CER        | 12.992637634277344   |
| test/DER        | 1.7323516607284546   |
| test/IER        | 2.9883065223693848   |
| test/SER        | 8.271979331970215    |
| test/loss       | 0.6014225482940674   |

## Result w/ Greedy Decoder

| Validate metric | DataLoader 0          |
|-----------------|----------------------|
| val/CER         | 11.940629005432129   |
| val/DER         | 1.794417381286621    |
| val/IER         | 2.9685423374176025   |
| val/SER         | 7.177669525146484    |
| val/loss        | 0.5399672389030457   |

| Test metric     | DataLoader 0          |
|-----------------|----------------------|
| test/CER        | 13.274145126342773   |
| test/DER        | 1.6890429258346558   |
| test/IER        | 3.2698137760162354   |
| test/SER        | 8.315287590026855    |
| test/loss       | 0.6014225482940674   |

- Max epochs = 100
- Best checkpoint at epoch = 88
- Num transformer layers = 5
- Dropout = 0.1
- Learning rate = 0.001
- Batch size = 16