

numpyknots = np.concatenate(([0,0,0],these_knots,[1,1,1])) # because??
y_py = np.zeros((x.shape[0], len(these_knots)+2))
for i in range(len(these_knots)+2):
    y_py[:,i] = intrp.BSpline(numpyknots, (np.arange(len(these_knots)+2)==i).astype(float), 3, extrapolate=False)(x)
