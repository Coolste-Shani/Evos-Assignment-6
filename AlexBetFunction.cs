using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.utils;

namespace CNNFunctions
{
    public class AlexNet : nn.Module<Tensor, Tensor>
    {
        //Place submodule references here
        private nn.Module<Tensor, Tensor> con1;
        private nn.Module<Tensor, Tensor> pool1;
        private nn.Module<Tensor, Tensor> con2;
        private nn.Module<Tensor, Tensor> pool2;
        private nn.Module<Tensor, Tensor> con3;
        private nn.Module<Tensor, Tensor> con4;
        private nn.Module<Tensor, Tensor> con5;
        private nn.Module<Tensor, Tensor> pool3;
        private nn.Module<Tensor, Tensor> lin1;
        private nn.Module<Tensor, Tensor> lin2;
        private nn.Module<Tensor, Tensor> lin3;
        private nn.Module<Tensor, Tensor> dropout;

        public AlexNet(int NrOutputs) : base(nameof(AlexNet)) //Call base constructor
        {
            //Instantiate submodules
            //Input 3x224x224
            con1 = nn.Conv2d(3, 96, kernel_size: 11, stride: 4); //Output 96x54x54
            pool1 = nn.MaxPool2d(3, stride: 2); //Output 96x26x26
            con2 = nn.Conv2d(96, 256, kernel_size: 5, stride: 1, padding: 2); //Output 256x26x26
            pool2 = nn.MaxPool2d(3, stride: 2); //Output 256x12x12
            con3 = nn.Conv2d(256, 384, kernel_size: 3, stride: 1, padding: 1);   //Output 384x12x12
            con4 = nn.Conv2d(384, 384, kernel_size: 3, stride: 1, padding: 1);   //Output 384x12x12
            con5 = nn.Conv2d(384, 256, kernel_size: 3, stride: 1, padding: 1);   //Output 256x12x12
            pool3 = nn.MaxPool2d(3, stride: 2); //Output 256x5x5
            lin1 = nn.Linear(6400, 1000);   //Originally nn.Linear(6400, 4096);
            dropout = nn.Dropout(0.1);
            lin2 = nn.Linear(1000, 250);    //Originally nn.Linear(4096, 4096);            
            lin3 = nn.Linear(250, NrOutputs);   //Originally nn.Linear(4096, NrOutputs);
            RegisterComponents();  //Call this to include submodule parameters in module parameters
        }

        public override Tensor forward(Tensor input)
        {
            var y = con1.forward(input);
            y = nn.functional.leaky_relu(y);
            y = pool1.forward(y);
            y = con2.forward(y);
            y = nn.functional.leaky_relu(y);
            y = pool2.forward(y);
            y = con3.forward(y);
            y = nn.functional.leaky_relu(y);
            y = con4.forward(y);
            y = nn.functional.leaky_relu(y);
            y = con5.forward(y);
            y = nn.functional.leaky_relu(y);
            y = pool3.forward(y);
            y = lin1.forward(torch.flatten(y, start_dim: 1));
            y = nn.functional.leaky_relu(y);
            y = dropout.forward(y);
            y = lin2.forward(y);
            y = nn.functional.leaky_relu(y);
            y = dropout.forward(y);
            y = lin3.forward(y);
            return nn.functional.softmax(y, dim: 1);
        }
    }
}
