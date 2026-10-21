"""Compact layer specification derived from the executed Net implementation."""
import os
def specification():
    linear=lambda a,b:(a+1)*b
    rows=[
        ['Shared inputs','X; F; coordinates','N × 9; N × 5; N × 2','—'],
        ['Spatial aggregation','Distance-weighted 8NN','N × 9','0'],
        ['Coordinate FC1','Linear 2 → 48; Tanh','N × 48',linear(2,48)],
        ['Coordinate FC2','Linear 48 → 48; Tanh','N × 48',linear(48,48)],
        ['Coefficient head','Linear 48 → 10','N × 10',linear(48,10)],
        ['Coordinate residual','Σ beta_j X_j + intercept','N × 1','0'],
        ['PGCAN context input','Concat [X, F, AX]','N × 23','0'],
        ['PGCAN context FC1','Linear 23 → 48; ReLU','N × 48',linear(23,48)],
        ['PGCAN context FC2','Linear 48 → 24; ReLU','N × 24',linear(48,24)],
        ['PGCAN context head','Linear 24 → 1','N × 1',linear(24,1)],
        ['Relation encoders¹','3 × (Linear 9 → 16; ReLU)','N × 3 × 16',3*linear(9,16)],
        ['Relation gate¹','Linear [X,F,s]: 16 → 3; softmax','N × 3',linear(16,3)],
        ['Relation fusion¹','Gate-weighted sum','N × 16','0'],
        ['MR context input¹','Concat [X, F, AX, fusion]','N × 39','0'],
        ['MR context FC1¹','Linear 39 → 48; ReLU','N × 48',linear(39,48)],
        ['MR context FC2¹','Linear 48 → 24; ReLU','N × 24',linear(48,24)],
        ['MR context head¹','Linear 24 → 1','N × 1',linear(24,1)],
        ['Allocation (both)','softmax(log(max(p0, ε)) + r)','N × 1; sum = 1','0'],
        ['Total registered','PGCAN / multi-relation (MR)','—','5,339 / 6,638'],
        ['Optimizer','Adam; learning rate 0.008','Weight decay 0.0002','—'],
        ['Batch / gradient','Full training-node batch','Gradient norm cap 5','—'],
        ['Loss weights','Prior / graph / coefficient','0.025 / 0.015 / 0.0005','—'],
        ['Spatial refits','5 outer folds; 3 seeds','Up to 450 epochs','—'],
        ['Saved original','5 seeds; no early stopping','900 epochs','—']]
    shared=linear(2,48)+linear(48,48)+linear(48,10)
    assert shared+linear(23,48)+linear(48,24)+linear(24,1)==5339
    assert shared+3*linear(9,16)+linear(16,3)+linear(39,48)+linear(48,24)+linear(24,1)==6638
    return ['Block / layer','Operator','Output tensor / setting','Parameters'],[[str(x) for x in row] for row in rows],[43,63,44,24]
