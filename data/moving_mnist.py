import socket
import numpy as np
from torchvision import datasets, transforms
import pdb
import matplotlib.pyplot as plt

class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=True):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 

        self.data = datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Scale(self.digit_size),
                 transforms.RandomRotation(degrees=(90, -90), fill=(0,)),                 #only during testing
                 transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),    #only during testing
                 transforms.ToTensor()]))

        self.N = len(self.data) 

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      self.channels),
                    dtype=np.float32)

        examples = enumerate(self.data)
        batch_idx, (example_data, example_targets) = next(examples)
        print(example_data.shape)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            print(self.N)
            digit, label = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):

                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)
                   
                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                sy += dy    
                sx += dx
        #         print(x.shape)
                xx=np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                noise = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
                x_noisy = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                x_zeros = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                
                x_noisy = (x[t] + noise)
                # plt.imshow(x_noisy[t,:,:,0],cmap='gray')
                # plt.show()
                # if t <3:
                #     x_noisy = (x[t] + noise)
                #     #plt.imshow(x_noisy[t,:,:,0],cmap='gray')
                #     #plt.show()
                #     #x_noisy[t, :,:, 0] += digit.numpy().squeeze()
                # else:
                #     x_noisy = (x[t] + x_zeros)
                    #plt.imshow(x_noisy[t,:,:,0],cmap='gray')
                    #plt.show()

                    #x_noisy[t, :,:, 0] += digit.numpy().squeeze()
                #plt.imshow(x_noisy[t, :,:,0],cmap ='gray')
            #print('x shape',x.shape)
            #print('x_noisy',x_noisy.shape)
                # plt.imshow(x_noisy[t,:,:,0],cmap='gray')
                # plt.show()
        x_noisy[x_noisy>1] = 1
        #x[x>1] = 1
        #print("n value", self.N)
        #print('x_noisy shape',x_noisy.shape)
        return x_noisy
        # plt.imshow(x_noisy[0,:,:,0],cmap='gray')
        # plt.show()
        # print(x_noisy.shape)
        # print(x_noisy.shape)
        # print(x.shape)
        # print(x_zeros.shape)
        # for i in range(8):
        #     if i<3:
        #         x_noisy = (x[i] + noise).numpy().squeeze
        #         # plt.imshow(xx[0,:,:,0], cmap='gray')
        #         # plt.show()
                
        #     else:
        #         x_noisy = (x[i] + x_zeros).numpy().squeeze
        #         xx.append(x_noisy)

    #print(out.shape)
        # plt.imshow(x_noisy[1,:,:,0], cmap='gray')
        # plt.show()

        #x_noisy[x_noisy>1] = 1
        

# def main():
    
#     train_data = MovingMNIST(
#         train=True,
#         data_root='data',
#         seq_len=8,
#         image_size=64,
#         deterministic=True,
#         num_digits=1)
#     #N = len(test_data)
#     #print(N)

#     out = train_data[0]
#     # noise = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
#     # x_noisy = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
#     # x_zeros = np.zeros((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

#     # for i in range(seq_len):
#     #     if i<3:
#     #         x_noisy[i] = x + noise
#     #     else:
#     #         x_noisy[i] = x + x_zeros

#     # #print(out.shape)
#     # plt.imshow(x_noisy[i], cmap='gray')
#     # plt.show()

# if __name__ == '__main__' :
#     main()

