import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

# Over Fit Model
class OverFitModel(torch.nn.Module):
    def __init__(self):
        super(OverFitModel, self).__init__()
        self.linear = torch.nn.Linear(5, 1) # One in and one out

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    # all data
    train_x_plot_u = np.arange(0, 5 + 1, 0.01)
    train_x_plot_o = np.array([[e, e ** 2,  e**3,  e**4,  e**5] for e in train_x_plot_u])
    train_x_u = np.arange(0, 5 + 1, 1)
    train_x_o = np.array([[e, e ** 2,  e**3,  e**4,  e**5] for e in train_x_u])
    # train_noise = -3 + 6 * np.random.rand(len(train_x_u)) 
    train_noise = np.array([-2.06345651,  0.06141883, -2.26895645, -2.03486411,  1.28754263, -2.5007614 ])
    train_y =  (train_x_u - 2.5 ) ** 2 + 7 + train_noise
    
    test_x_plot_u = np.arange(6, 9 + 1, 0.01)
    test_x_plot_o = np.array([[e, e ** 2,  e**3,  e**4,  e**5] for e in test_x_plot_u])
    test_x_u = np.arange(6, 9 + 1, 1)
    test_x_o = np.array([[e, e ** 2,  e**3,  e**4,  e**5] for e in test_x_u])
    test_y =  (test_x_u - 2.5 ) ** 2 + 7

    # numpy to torch.tensor
    train_tensor_x_plot_u = torch.Tensor([[e] for e in train_x_plot_u]) # 将1维的数据转换为2维数据
    train_tensor_x_plot_o = torch.Tensor(train_x_plot_o) # 将1维的数据转换为2维数据
    train_tensor_x_u= torch.Tensor([[e] for e in train_x_u]) # 将1维的数据转换为2维数据
    train_tensor_x_o= torch.Tensor(train_x_o) # 将1维的数据转换为2维数据
    train_tensor_y = torch.Tensor([[e] for e in train_y])


    test_tensor_x_plot_u = torch.Tensor([[e] for e in test_x_plot_u]) # 将1维的数据转换为2维数据
    test_tensor_x_plot_o = torch.Tensor(test_x_plot_o) # 将1维的数据转换为2维数据
    test_tensor_x_u= torch.Tensor([[e] for e in test_x_u]) # 将1维的数据转换为2维数据
    test_tensor_x_o= torch.Tensor(test_x_o) # 将1维的数据转换为2维数据
    test_tensor_y = torch.Tensor([[e] for e in test_y])

    n_train_time_over_fit = 240000
    # n_train_time_over_fit = 10


    

    y_lim = [0,  20]
   
    # Over Fit Train 
    over_fit_model = OverFitModel()
    loss_fun = torch.nn.MSELoss() # Defined loss function
    over_fit_opt = torch.optim.Adam(over_fit_model.parameters(), lr=0.01) # Defined optimizer
    for epoch in range(n_train_time_over_fit):
        tmp_pred_tensor_y = over_fit_model(train_tensor_x_o) # Forward pass
        loss = loss_fun(tmp_pred_tensor_y, train_tensor_y)  # Compute loss
        if epoch % (n_train_time_over_fit/10) == 0:
            print(epoch, loss.data.numpy())
        over_fit_opt.zero_grad()  # Zero gradients  
        loss.backward() # perform backward pass
        over_fit_opt.step() # update weights
    over_fit_train_y = over_fit_model(train_tensor_x_o)
    loss_over_fit_train = loss_fun(over_fit_train_y, train_tensor_y).data.numpy()
    over_fit_plot_train_y = over_fit_model(train_tensor_x_plot_o)

    over_fit_test_y = over_fit_model(test_tensor_x_o)
    loss_over_fit_test = loss_fun(over_fit_test_y, test_tensor_y).data.numpy()
    over_fit_plot_test_y = over_fit_model(test_tensor_x_plot_o) 
    print("Done over fit")







    # L1 normlization Fit Train 
    over_fit_model = OverFitModel()
    loss_fun = torch.nn.MSELoss()
    l1_lambda = torch.tensor(1000.)

    # over_fit_opt = torch.optim.Adam(over_fit_model.parameters(), lr=0.01, weight_decay=10) # With L2 normlization
    over_fit_opt = torch.optim.Adam(over_fit_model.parameters(), lr=0.01) # With L2 normlization
    for epoch in range(n_train_time_over_fit):
        tmp_pred_tensor_y = over_fit_model(train_tensor_x_o) # Forward pass
        loss = loss_fun(tmp_pred_tensor_y, train_tensor_y)  # Compute loss

        # use L1 normlization
        l1_loss = torch.tensor(0.)
        for param in over_fit_model.parameters():
            l1_loss += torch.sum(torch.abs(param))
        loss += l1_lambda * l1_loss

        if epoch % (n_train_time_over_fit/10) == 0:
            print(epoch, loss.data.numpy())
        over_fit_opt.zero_grad()  # Zero gradients  
        loss.backward() # perform backward pass
        over_fit_opt.step() # update weights
    l2_train_y = over_fit_model(train_tensor_x_o)
    loss_l2_train = loss_fun(l2_train_y, train_tensor_y).data.numpy()
    l2_plot_train_y = over_fit_model(train_tensor_x_plot_o)

    l2_test_y = over_fit_model(test_tensor_x_o)
    loss_l2_test = loss_fun(l2_test_y, test_tensor_y).data.numpy()
    l2_plot_test_y = over_fit_model(test_tensor_x_plot_o) 
    print("Done L1 normalization ")



    plt.figure()

    plt.subplot(2,2,1)      
    plt.title("Over Fit(Train Set), Loss=%.3f" % loss_over_fit_train) 
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.scatter(train_x_u, train_y)
    plt.plot(train_x_plot_u, over_fit_plot_train_y.data.numpy(), color='red')
    plt.ylim(y_lim)
    for a, b in zip(train_x_u, train_y):  
        plt.text(a, b,  np.round(b, 1),ha='center', va='bottom', fontsize=10)  

    
    plt.subplot(2,2,2)      
    plt.title("L1 normlization(Train Set), Loss=%.3f" % loss_l2_train) 
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.scatter(train_x_u, train_y)
    plt.plot(train_x_plot_u, l2_plot_train_y.data.numpy(), color='orange')
    plt.ylim(y_lim)
    for a, b in zip(train_x_u, train_y):  
        plt.text(a, b,  np.round(b, 1),ha='center', va='bottom', fontsize=10)  




    plt.subplot(2,2,3)      
    plt.title("Over Fit(Test Set), Loss=%.3f" % loss_over_fit_test) 
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.scatter(test_x_u, test_y)
    plt.plot(test_x_plot_u, over_fit_plot_test_y.data.numpy(), color='red')
    plt.ylim([-1000,1000])
    for a, b in zip(test_x_u, test_y):  
        plt.text(a, b,  np.round(b, 1),ha='center', va='bottom', fontsize=10)  
    
    plt.subplot(2,2,4)      
    plt.title("L1 normlization (Test Set), Loss=%.3f" % loss_l2_test) 
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.scatter(test_x_u, test_y)
    plt.plot(test_x_plot_u, l2_plot_test_y.data.numpy(), color='orange')
    plt.ylim([0, 100])
    for a, b in zip(test_x_u, test_y):  
        plt.text(a, b,  np.round(b, 1),ha='center', va='bottom', fontsize=10)  
    plt.show()