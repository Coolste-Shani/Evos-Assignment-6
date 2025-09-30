using CNNFunctions;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

class Program
{
    static void Main()
    {
        new Program();
    }

    Program()
    {
        //AlexNetArch();
        ResNetArch();
    }
    public void AlexNetArch()
    {
        ImageDataLoaderxx data;
        float eta = 0.01f;//0.00001f;//0.01f;//
        Console.WriteLine("Loading Images....");
        data = new ImageDataLoaderxx();
        //data.LoadAndSave();

        data.LoadFromSaved();
        Console.WriteLine("Image Loading Complete");
        var tIndexed = torch.argmax(data.t, 1);       //Remove one-hot encoding
        var ttestIndexed = torch.argmax(data.ttest, 1); //Remove one-hot encoding

        Console.WriteLine(data.z.ToString());
        Console.WriteLine(data.t.ToString());
        Console.WriteLine(data.ztest.ToString());
        Console.WriteLine(data.ttest.ToString());

        var mynn = new AlexNet(data.NrClasses);
        var loss = nn.NLLLoss();
        var optimizer = torch.optim.SGD(mynn.parameters(), eta);//torch.optim.Adam(mynn.parameters(), eta);//torch.optim.RMSProp(mynn.parameters(), eta);//
        mynn.eval();
        var validerror = loss.forward(mynn.forward(data.ztest).log(), ttestIndexed);
        mynn.train();
        for (int g = 0; g <= 1000; g++)
        {
            int bR = (g * 500) % (int)data.z.shape[0];  //Begin range
            int eR = Math.Min(bR + 600, (int)data.z.shape[0]);  //End range
            var nnoutput = mynn.forward(data.z[bR..eR]);
            var error = loss.forward(nnoutput.log(), tIndexed[bR..eR]);
            if (g % 10 == 0) Console.WriteLine(g + " " + error.item<float>());

            mynn.zero_grad();
            error.backward();
            optimizer.step();

            if (g % 50 == 0)
            {
                mynn.eval();
                var tempvaliderror = loss.forward(mynn.forward(data.ztest).log(), ttestIndexed);
                Console.WriteLine("TestError: " + tempvaliderror.item<float>());
                if (validerror.item<float>() > tempvaliderror.item<float>())
                {
                    mynn.save("MySecondCNN.mc");
                    validerror = tempvaliderror;
                    Console.WriteLine("Saved " + g);
                }

                mynn.train();
            }
            System.GC.Collect();
        }
        Console.WriteLine("Training Complete");

        var trainednn = new AlexNet(data.NrClasses);
        trainednn.load("MySecondCNN.mc");
        trainednn.eval();
        //Test network on test set
        var tsse = torch.tensor(0.0f);
        int[] classErrors = new int[data.NrClasses];
        var output = trainednn.forward(data.ztest);
        for (int p = 0; p < data.TestSetSize; p++)
        {
            var error = ((output[p] - data.ttest[p]) * (output[p] - data.ttest[p])).sum();
            tsse.add_(error);

            if (torch.argmax(output[p]).item<long>() != torch.argmax(data.ttest[p]).item<long>())
            {
                classErrors[torch.argmax(data.ttest[p]).item<long>()]++;
            }

            Console.WriteLine("Expected Outputs: " + data.ttest[p].ToString(torch.numpy));
            Console.WriteLine("Outputs: " + output[p].ToString(torch.numpy));
        }
        Console.WriteLine("SSE on test set: " + tsse.ToString(torch.numpy));
        int totalerrors = 0;
        for (int i = 0; i < data.NrClasses; i++)
        {
            Console.WriteLine("Errors for class " + i + " is " + classErrors[i]);
            totalerrors += classErrors[i];
        }
        Console.WriteLine("Total errors " + totalerrors + " / " + data.TestSetSize);
    }

    public void ResNetArch()
    {
        ImageDataLoaderxx data;
        float eta = 0.01f;//0.00001f;//0.01f;//
        Console.WriteLine("Loading Images....");
        data = new ImageDataLoaderxx();
        //data.LoadAndSave();

        data.LoadFromSaved();
        Console.WriteLine("Image Loading Complete");
        var tIndexed = torch.argmax(data.t, 1);       //Remove one-hot encoding
        var ttestIndexed = torch.argmax(data.ttest, 1); //Remove one-hot encoding

        Console.WriteLine(data.z.ToString());
        Console.WriteLine(data.t.ToString());
        Console.WriteLine(data.ztest.ToString());
        Console.WriteLine(data.ttest.ToString());

        var mynn = new ResNet(data.NrClasses);
        var loss = nn.NLLLoss();
        var optimizer = torch.optim.SGD(mynn.parameters(), eta);//torch.optim.Adam(mynn.parameters(), eta);//torch.optim.RMSProp(mynn.parameters(), eta);//
        mynn.eval();
        var validerror = loss.forward(mynn.forward(data.ztest).log(), ttestIndexed);
        mynn.train();
        for (int g = 0; g <= 1000; g++)
        {
            int bR = (g * 500) % (int)data.z.shape[0];  //Begin range
            int eR = Math.Min(bR + 600, (int)data.z.shape[0]);  //End range
            var nnoutput = mynn.forward(data.z[bR..eR]);
            var error = loss.forward(nnoutput.log(), tIndexed[bR..eR]);
            if (g % 10 == 0) Console.WriteLine(g + " " + error.item<float>());

            mynn.zero_grad();
            error.backward();
            optimizer.step();

            if (g % 50 == 0)
            {
                mynn.eval();
                var tempvaliderror = loss.forward(mynn.forward(data.ztest).log(), ttestIndexed);
                Console.WriteLine("TestError: " + tempvaliderror.item<float>());
                if (validerror.item<float>() > tempvaliderror.item<float>())
                {
                    mynn.save("MySecondCNN.mc");
                    validerror = tempvaliderror;
                    Console.WriteLine("Saved " + g);
                }

                mynn.train();
            }
            System.GC.Collect();
        }
        Console.WriteLine("Training Complete");

        var trainednn = new ResNet(data.NrClasses);
        trainednn.load("MySecondCNN.mc");
        trainednn.eval();
        //Test network on test set
        var tsse = torch.tensor(0.0f);
        int[] classErrors = new int[data.NrClasses];
        var output = trainednn.forward(data.ztest);
        for (int p = 0; p < data.TestSetSize; p++)
        {
            var error = ((output[p] - data.ttest[p]) * (output[p] - data.ttest[p])).sum();
            tsse.add_(error);

            if (torch.argmax(output[p]).item<long>() != torch.argmax(data.ttest[p]).item<long>())
            {
                classErrors[torch.argmax(data.ttest[p]).item<long>()]++;
            }

            Console.WriteLine("Expected Outputs: " + data.ttest[p].ToString(torch.numpy));
            Console.WriteLine("Outputs: " + output[p].ToString(torch.numpy));
        }
        Console.WriteLine("SSE on test set: " + tsse.ToString(torch.numpy));
        int totalerrors = 0;
        for (int i = 0; i < data.NrClasses; i++)
        {
            Console.WriteLine("Errors for class " + i + " is " + classErrors[i]);
            totalerrors += classErrors[i];
        }
        Console.WriteLine("Total errors " + totalerrors + " / " + data.TestSetSize);
    }

}









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
public class ImageDataLoader
{
    public torch.Tensor z;          //Training set inputs
    public torch.Tensor t;          //Training set expected outputs
    public torch.Tensor ztest;      //Test set inputs
    public torch.Tensor ttest;      //Test set expected outputs

    public int IWidth = 224;
    public int IHeight = 224;
    public int NrClasses = 2;
    public int TrainSetSize = 0;
    public int TestSetSize = 40;

    public void LoadAndSave()
    {
        string[] classFolders = new string[]
        {
              @"C:\Users\th4te\OneDrive - Nelson Mandela University\A-NMU\4th Year\Semester 2\WRCV402\Assignment_6\human",
              @"C:\Users\th4te\OneDrive - Nelson Mandela University\A-NMU\4th Year\Semester 2\WRCV402\Assignment_6\non_human"
        };

        for (int i = 0; i < NrClasses; i++)
        {
            String[] files = Directory.GetFiles(classFolders[i]);
            TrainSetSize += files.Length;
        }
        TrainSetSize -= TestSetSize;
        z = torch.empty(TrainSetSize, 3, IWidth, IHeight);
        t = torch.empty(TrainSetSize, NrClasses);
        ztest = torch.empty(TestSetSize, 3, IWidth, IHeight);
        ttest = torch.empty(TestSetSize, NrClasses);

        int zindex = 0;
        int tindex = 0;
        for (int i = 0; i < NrClasses; i++)
        {
            String[] files = Directory.GetFiles(classFolders[i]);
            int NrTrainImages = files.Length - (TestSetSize / NrClasses);

            for (int p = 0; p < NrTrainImages; p++)
            {
                z[zindex] = MCToolkit.LoadImage(files[p], IWidth, IHeight);

                var ttemp = torch.zeros(NrClasses);
                ttemp[i] = 1.0f;
                t[zindex] = ttemp;

                zindex++;
            }

            for (int p = NrTrainImages; p < files.Length; p++)
            {
                ztest[tindex] = MCToolkit.LoadImage(files[p], IWidth, IHeight);

                var ttemp = torch.zeros(NrClasses);
                ttemp[i] = 1.0f;
                ttest[tindex] = ttemp;

                tindex++;
            }
        }
        torch.save(z, @"C:\Users\th4te\source\repos\AA66\Data\ZData.pt");
        torch.save(t, @"C:\Users\th4te\source\repos\AA66\Data\TData.pt");
        torch.save(ztest, @"C:\Users\th4te\source\repos\AA66\Data\ZTData.pt");
        torch.save(ttest, @"C:\Users\th4te\source\repos\AA66\Data\TTData.pt");
        Console.WriteLine("Saved ");
    }

    public void LoadFromSaved()
    {
        z = torch.load("C:\\Users\\th4te\\source\\repos\\AA66\\Data\\ZData.pt");
        t = torch.load("C:\\Users\\th4te\\source\\repos\\AA66\\Data\\TData.pt");
        ztest = torch.load("C:\\Users\\th4te\\source\\repos\\AA66\\Data\\ZTData.pt");
        ttest = torch.load("C:\\Users\\th4te\\source\\repos\\AA66\\Data\\TTData.pt");
    }
}

public class MCToolkit
{
    public static (torch.Tensor a, torch.Tensor b) Shuffle(torch.Tensor Z, torch.Tensor T)
    {
        long[] shZ = Z.shape;
        long[] shT = T.shape;
        Z = torch.flatten(Z, 1);
        T = torch.flatten(T, 1);

        var perms = torch.randperm(shZ[0]);
        var M = torch.zeros(shZ[0], shZ[0]);
        for (int i = 0; i < shZ[0]; i++)
        {
            M[i, perms[i]] = 1.0f;
        }

        var Zn = M.mm(Z);
        var Tn = M.mm(T);

        Zn = Zn.reshape(shZ);
        Tn = Tn.reshape(shT);

        return (Zn, Tn);
    }

    public static torch.Tensor LoadImage(string filename, int IWidth, int IHeight)
    {
        Bitmap bm = new Bitmap(filename);

        if ((bm.Height != IHeight) || (bm.Width != IWidth)) Console.WriteLine("Incorrect Image Size");

        float[] rgbPixels = new float[3 * bm.Width * bm.Height];

        int i = 0;
        for (int y = 0; y < bm.Height; y++)
        {
            for (int x = 0; x < bm.Width; x++)
            {
                Color rgb = bm.GetPixel(x, y);

                // RGB values scaled between 0 and 1
                rgbPixels[i] = rgb.R / 255.0f;
                rgbPixels[i + bm.Height * bm.Width] = rgb.G / 255.0f;
                rgbPixels[i + 2 * +bm.Height * bm.Width] = rgb.B / 255.0f;
                i++;
            }
        }

        return torch.tensor(rgbPixels, dtype: ScalarType.Float32).reshape(3, bm.Height, bm.Width);
    }

    public static void SaveImage(string filename, Tensor I)
    {
        Bitmap bm = MakeImage(I);
        bm.Save(filename, ImageFormat.Bmp);

    }

    public static Bitmap MakeImage(Tensor I)
    {
        long[] IShape = I.shape;

        Bitmap bm = new Bitmap((int)IShape[2], (int)IShape[1]);

        for (int y = 0; y < bm.Height; y++)
        {
            for (int x = 0; x < bm.Width; x++)
            {
                Color rgb = Color.FromArgb((int)(I[0][y][x] * 255), (int)(I[1][y][x] * 255), (int)(I[2][y][x] * 255));
                bm.SetPixel(x, y, rgb);
            }
        }
        return bm;

    }
}