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


def display_specification():
    """Sensen-inspired compact presentation; parameter ledger stays separate."""
    rows=[
        ['Coordinate branch (both)','','',''],
        ['Input: coordinates s','','2','N × 2'],
        ['FC1 + Tanh','2','48','N × 48'],
        ['FC2 + Tanh','48','48','N × 48'],
        ['Coefficient head','48','10','N × 10'],
        ['PGCAN context','','',''],
        ['Input: X, F, AX','','23','N × 23'],
        ['FC1 + ReLU','23','48','N × 48'],
        ['FC2 + ReLU','48','24','N × 24'],
        ['Output','24','1','N × 1'],
        ['Multi-relation context','','',''],
        ['3 relation encoders + ReLU','9','16','N × 3 × 16'],
        ['Gate + softmax','16','3','N × 3'],
        ['Weighted fusion','3 × 16','16','N × 16'],
        ['Input: X, F, AX, h','','39','N × 39'],
        ['FC1 + ReLU','39','48','N × 48'],
        ['FC2 + ReLU','48','24','N × 24'],
        ['Output','24','1','N × 1'],
        ['Hyperparameters','Value','Protocol','Value'],
        ['Optimizer','Adam','Learning rate','0.008'],
        ['Weight decay','0.0002','Gradient cap','5'],
        ['PGCAN parameters','5,339','MR parameters','6,638'],
        ['Training batch','Full nodes','Loss','CE + penalties'],
        ['Prior / graph / coefficient','0.025 / 0.015 / 0.0005','',''],
        ['Original fit','900 epochs','Seeds','5'],
        ['Spatial refit','≤450 epochs','Seeds','3'],
        ['Outer folds / buffer','5 / 1 km','Block size','10 km']]
    return ['Network layer','Input width','Output width','Output tensor'],rows,[66,29,29,50]


SECTION_ROWS={0,5,10,18}
MERGED_VALUE_ROW=23
DISPLAY_NOTE=('N is the current node count; X, F and AX have 9, 5 and 9 channels. '
    'The 10 coordinate outputs comprise nine coefficients and one intercept. '
    'The MR gate receives [X, s, F] (16 inputs); the fused vector h has 16 channels. '
    'Both models add coordinate and context residuals before the prior-anchored allocation softmax. '
    'Counts include biases; MR replaces the PGCAN context head. Loss entries list the prior, graph and coefficient penalty weights. '
    'Original full-domain fits and buffered spatial refits use the separate epoch/seed settings shown. '
    'Inner epoch selection is checked every 15 epochs, with patience five checks and a minimum of 120 epochs. '
    'There are no convolution, batch-normalization or dropout layers. '
    'The exact per-layer parameter ledger is supplied as Architecture_parameter_ledger.csv.')
