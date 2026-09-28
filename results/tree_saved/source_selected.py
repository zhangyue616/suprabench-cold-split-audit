DESC = [('MW', Descriptors.MolWt), ('LogP', Descriptors.MolLogP), ('TPSA', Descriptors.TPSA), ('HBD', Descriptors.NumHDonors), ('HBA', Descriptors.NumHAcceptors), ('RotB', Descriptors.NumRotatableBonds), ('Ring', rdMolDescriptors.CalcNumRings), ('ArRing', rdMolDescriptors.CalcNumAromaticRings), ('FrCSP3', Descriptors.FractionCSP3), ('Heavy', lambda m: m.GetNumHeavyAtoms()), ('Charge', Chem.GetFormalCharge)]

def dvec(s):
    m = Chem.MolFromSmiles(s)
    o = []
    for _, fn in DESC:
        try:
            o.append(float(fn(m)))
        except Exception:
            o.append(0.0)
    return np.array(o, np.float32)
uh = {s: dvec(s) for s in df['host_smiles'].unique()}
ug = {s: dvec(s) for s in df['guest_smiles'].unique()}
H = np.vstack([uh[s] for s in df['host_smiles']])
G = np.vstack([ug[s] for s in df['guest_smiles']])

def sdiv(a, b):
    return a / np.where(np.abs(b) < 1e-06, 1e-06, b)
P = np.column_stack([H[:, 0] + G[:, 0], sdiv(G[:, 0], H[:, 0]), H[:, 9] - G[:, 9], sdiv(G[:, 9], H[:, 9]), H[:, 10] * G[:, 10], H[:, 10] + G[:, 10], np.abs(H[:, 2] - G[:, 2]), H[:, 3] + G[:, 4], H[:, 4] + G[:, 3], H[:, 1] + G[:, 1]]).astype(np.float32)
X_pair = np.hstack([H, G, P])

def ecfp(s):
    return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024), np.float32)
he = {s: ecfp(s) for s in df['host_smiles'].unique()}
ge = {s: ecfp(s) for s in df['guest_smiles'].unique()}
X_ecfp = np.hstack([np.vstack([he[s] for s in df['host_smiles']]), np.vstack([ge[s] for s in df['guest_smiles']])])

def guest3d(s):
    try:
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumHeavyAtoms() > 70:
            return None
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3()
        p.randomSeed = 0
        p.useRandomCoords = True
        p.maxIterations = 200
        if AllChem.EmbedMolecule(m, p) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
        return np.array([rdMolDescriptors.CalcAsphericity(m), rdMolDescriptors.CalcEccentricity(m), rdMolDescriptors.CalcInertialShapeFactor(m), rdMolDescriptors.CalcNPR1(m), rdMolDescriptors.CalcNPR2(m), rdMolDescriptors.CalcRadiusOfGyration(m), rdMolDescriptors.CalcSpherocityIndex(m), rdMolDescriptors.CalcPMI1(m), rdMolDescriptors.CalcPMI2(m), rdMolDescriptors.CalcPMI3(m)], np.float32)
    except Exception:
        return None

def rf(X, tr, te, seed):
    return RandomForestRegressor(300, n_jobs=-1, random_state=seed, max_features='sqrt').fit(X[tr], y[tr]).predict(X[te])

def xgbr(X, tr, te, seed):
    return xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, random_state=seed, tree_method='hist').fit(X[tr], y[tr]).predict(X[te])

def lgbr(X, tr, te, seed):
    return lgb.LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, subsample_freq=1, reg_lambda=1.0, n_jobs=-1, random_state=seed, verbose=-1).fit(X[tr], y[tr]).predict(X[te])
