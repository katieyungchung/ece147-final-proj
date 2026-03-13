## Best Result w/ Beam Decoder

| Validate metric | DataLoader 0       |
| --------------- | ------------------ |
| val/CER         | 10.456357955932617 |
| val/DER         | 1.5285778045654297 |
| val/IER         | 2.303943395614624  |
| val/SER         | 6.623836994171143  |
| val/loss        | 0.581344485282898  |

| Test metric | DataLoader 0       |
| ----------- | ------------------ |
| test/CER    | 12.256387710571289 |
| test/DER    | 1.5807708501815796 |
| test/IER    | 2.9883065223693848 |
| test/SER    | 7.687310695648193  |
| test/loss   | 0.623360812664032  |

## Result w/ Greedy Decoder

| Validate metric | DataLoader 0       |
| --------------- | ------------------ |
| val/CER         | 10.522817611694336 |
| val/DER         | 1.5285778045654297 |
| val/IER         | 2.4590163230895996 |
| val/SER         | 6.535223960876465  |
| val/loss        | 0.5813445448875427 |

| Test metric | DataLoader 0       |
| ----------- | ------------------ |
| test/CER    | 12.321351051330566 |
| test/DER    | 1.4941532611846924 |
| test/IER    | 3.1831963062286377 |
| test/SER    | 7.6440019607543945 |
| test/loss   | 0.6233608722686768 |

- Max epochs = 150
- Best checkpoint at epoch = 137
- Num transformer layers = 5
- Dropout = 0.1
- Learning rate = 0.001
- Batch size = 16