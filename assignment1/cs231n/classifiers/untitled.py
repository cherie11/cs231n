  m = X.shape[0]
  score = X.dot(W)
  score_mean = score - np.max(score, axis = 1).reshape(m, 1)
  score_exp = np.exp(score_mean)
  sum_row = np.sum(score_exp, 1)
  loss += -np.sum(np.log(score_exp[range(m), y]/sum_row))
  # gradient
  coef = score_exp / sum_row.reshape(m,1)
  coef[np.arange(m), y] -= 1
  dW = X.T.dot(coef)
  # regularization
  loss /= m
  loss +=  reg * np.sum(W * W)
  dW /= m
  dW += 2*reg * W