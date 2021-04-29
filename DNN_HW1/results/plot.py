# %% Code snippets
import matplotlib.pyplot as plt


def log():
    log = {"loss": [], "accuracy": []}

    log["loss"] += [loss]
    log["accuracy"] += [utils.Accuracy(y_predicted, y_batched)]

    return log


plt.figure(1)
plt.title("full-batch training")
plt.xlabel("epoch")
plt.xticks(np.arange(0, 330, step=30))
plt.ylabel("accuaracy")
plt.plot(log["accuracy"])
plt.legend(["train accuarcy"])

plt.figure(2)
plt.title("full-batch training")
plt.xlabel("epoch")
plt.xticks(np.arange(0, 330, step=30))
plt.ylabel("loss")
plt.plot(log["loss"])
plt.legend(["train loss"])