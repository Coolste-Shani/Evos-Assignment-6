using CNNFunctions;
using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using System.Drawing;


namespace CNNFunctions
{



    public class BasicBlock : nn.Module<Tensor, Tensor>
    {
        private nn.Module<Tensor, Tensor> conv1;
        private nn.Module<Tensor, Tensor> bn1;
        private nn.Module<Tensor, Tensor> conv2;
        private nn.Module<Tensor, Tensor> bn2;
        private nn.Module<Tensor, Tensor> shortcut;

        public BasicBlock(int inPlanes, int planes, int stride = 1) : base(nameof(BasicBlock))
        {
            conv1 = Conv2d(inPlanes, planes, kernel_size: 3, stride: stride, padding: 1, bias: false);
            bn1 = BatchNorm2d(planes);
            conv2 = Conv2d(planes, planes, kernel_size: 3, stride: 1, padding: 1, bias: false);
            bn2 = BatchNorm2d(planes);

            // Shortcut connection
            if (stride != 1 || inPlanes != planes)
            {
                shortcut = Sequential(
                    Conv2d(inPlanes, planes, kernel_size: 1, stride: stride, bias: false),
                    BatchNorm2d(planes)
                );
            }
            else
            {
                shortcut = Identity();
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor x)
        {
            var out1 = conv1.forward(x);
            out1 = bn1.forward(out1);
            out1 = functional.relu(out1);
            out1 = conv2.forward(out1);
            out1 = bn2.forward(out1);
            var shortcutOut = shortcut.forward(x);
            return functional.relu(out1 + shortcutOut);
        }
    }

    public class ResNet : nn.Module<Tensor, Tensor>
    {
        private nn.Module<Tensor, Tensor> conv1;
        private nn.Module<Tensor, Tensor> bn1;
        private nn.Module<Tensor, Tensor> layer1;
        private nn.Module<Tensor, Tensor> layer2;
        private nn.Module<Tensor, Tensor> layer3;
        private nn.Module<Tensor, Tensor> layer4;
        private nn.Module<Tensor, Tensor> fc;
        private nn.Module<Tensor, Tensor> avgpool;

        private int inPlanes = 64;

        public ResNet(int NrOutputs) : base(nameof(ResNet))
        {
            conv1 = Conv2d(3, 64, kernel_size: 7, stride: 2, padding: 3, bias: false);
            bn1 = BatchNorm2d(64);
            avgpool = AdaptiveAvgPool2d(1);

            layer1 = MakeLayer(64, 2, stride: 1);
            layer2 = MakeLayer(128, 2, stride: 2);
            layer3 = MakeLayer(256, 2, stride: 2);
            layer4 = MakeLayer(512, 2, stride: 2);

            fc = Linear(512, NrOutputs);

            RegisterComponents();
        }

        private nn.Module<Tensor, Tensor> MakeLayer(int planes, int blocks, int stride)
        {
            var layers = new List<nn.Module<Tensor, Tensor>>();

            layers.Add(new BasicBlock(inPlanes, planes, stride));
            inPlanes = planes;
            for (int i = 1; i < blocks; i++)
            {
                layers.Add(new BasicBlock(inPlanes, planes));
            }
            return Sequential(layers);
        }

        public override Tensor forward(Tensor x)
        {
            x = conv1.forward(x);
            x = bn1.forward(x);
            x = functional.relu(x);
            x = functional.max_pool2d(x, kernel_size: 3, stride: 2, padding: 1);

            x = layer1.forward(x);
            x = layer2.forward(x);
            x = layer3.forward(x);
            x = layer4.forward(x);

            x = avgpool.forward(x);
            x = torch.flatten(x, 1);
            x = fc.forward(x);
            return functional.softmax(x, dim: 1);
        }
    }


}
