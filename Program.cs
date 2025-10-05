//Instructions are also in the ReadMe file,
        //Before running the code,
        //Change the file paths that's in the DataLoaderxx functions to the corresponding filepaths of human and non human.
        //Uncomment the architecture to test
        // Run the code for the first time
        // The second time the code is ran, go to the architecture function and comment out the data.LoadAndSave(){ Reason is in the ReadMe file}
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
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
        ResNetArch();
    }

    public void ResNetArch()
    {
        ImageDataLoaderxx data;
        float eta = 0.1f;
        Console.WriteLine("Loading Images....");
        data = new ImageDataLoaderxx();
        //data.LoadAndSave();
        data.LoadFromSaved();
        int batchSize = 25;//25
        int epochs = 15;//8
        int batchesPerEpoch = (int)Math.Ceiling((double)data.z.shape[0] / batchSize);
        int totalSteps = batchesPerEpoch * epochs;
        Console.WriteLine("Image Loading Complete");

        var tIndexed = torch.argmax(data.t, 1);
        var ttestIndexed = torch.argmax(data.ttest, 1);
        var tvalidationIndexed = torch.argmax(data.tvalidation, 1);

        var mynn = new ResNet(data.NrClasses);
        var loss = CrossEntropyLoss();
        var optimizer = torch.optim.SGD(mynn.parameters(), eta);

        mynn.eval();
        var validerror = loss.forward(mynn.forward(data.zvalidation), tvalidationIndexed);
        mynn.train();

        for (int g = 0; g <= totalSteps; g++)
        {
            if (g % batchesPerEpoch == 0 && g != 0)
            {
                (data.z, data.t) = MCToolkit.Shuffle(data.z, data.t);
                tIndexed = torch.argmax(data.t, 1); // Update after shuffle
            }

            int bR = (g * batchSize) % (int)data.z.shape[0];
            int eR = Math.Min(bR + batchSize, (int)data.z.shape[0]);

            if (bR >= eR) continue; // Skip empty batches

            var nnoutput = mynn.forward(data.z[bR..eR]);
            var error = loss.forward(nnoutput, tIndexed[bR..eR]);

            if (g % batchesPerEpoch == 0)
                Console.WriteLine("Iteration: " + g + " Training error " + error.item<float>());

            mynn.zero_grad();
            error.backward();
            optimizer.step();

            if (g % batchesPerEpoch == 0 && g != 0)
            {
                mynn.eval();
                var tempvaliderror = loss.forward(mynn.forward(data.zvalidation), tvalidationIndexed);
                Console.WriteLine(" Validation Error: " + tempvaliderror.item<float>());
                if (validerror.item<float>() > tempvaliderror.item<float>())
                {
                    mynn.save("CurrentBest.mc");
                    validerror = tempvaliderror;
                    Console.WriteLine("Saved " + g);
                }
                mynn.train();
            }

        }
        Console.WriteLine("Training Complete");

        var trainednn = new ResNet(data.NrClasses);
        trainednn.load("CurrentBest.mc");
        trainednn.eval();

        // Test network on test set
        var output = trainednn.forward(data.ztest);
        var predictions = torch.argmax(output, 1);
        var actual = torch.argmax(data.ttest, 1);

        var correct = (predictions == actual).sum().item<long>();
        var accuracy = (double)correct / data.TestSetSize * 100.0;

        Console.WriteLine($"Overall Accuracy: {accuracy:F2}%");
        Console.WriteLine($"Correct: {correct}/{data.TestSetSize}");

        // Calculate per-class accuracy
        int humanCorrect = 0, humanTotal = 0;
        int nonHumanCorrect = 0, nonHumanTotal = 0;

        for (int p = 0; p < data.TestSetSize; p++)
        {
            var predictedClass = predictions[p].item<long>();
            var actualClass = actual[p].item<long>();

            if (actualClass == 0) // Human
            {
                humanTotal++;
                if (predictedClass == actualClass) humanCorrect++;
            }
            else // Non-Human
            {
                nonHumanTotal++;
                if (predictedClass == actualClass) nonHumanCorrect++;
            }
        }

        double humanAccuracy = humanTotal > 0 ? (double)humanCorrect / humanTotal * 100.0 : 0;
        double nonHumanAccuracy = nonHumanTotal > 0 ? (double)nonHumanCorrect / nonHumanTotal * 100.0 : 0;

        Console.WriteLine($"Correct Human: {humanAccuracy:F0}% {humanCorrect}/{humanTotal}");
        Console.WriteLine($"Correct Non Human: {nonHumanAccuracy:F0}% {nonHumanCorrect}/{nonHumanTotal}");

        // Show all test results
        Console.WriteLine("\nAll Test Results:");
        for (int p = 0; p < data.TestSetSize; p++)
        {
            var predictedProbs = torch.softmax(output[p], 0);
            var actualOneHot = data.ttest[p];

            Console.WriteLine($"Predicted: [{predictedProbs[0].item<float>():F4}, {predictedProbs[1].item<float>():F4}] || Actual: [{actualOneHot[0].item<float>()}, {actualOneHot[1].item<float>()}]");
        }
        //Console.WriteLine("Training Complete");

        //var trainednn = new ResNet(data.NrClasses);
        //trainednn.load("CurrentBest.mc");
        //trainednn.eval();

        //// Test network on test set
        //var output = trainednn.forward(data.ztest);
        //var predictions = torch.argmax(output, 1);
        //var actual = torch.argmax(data.ttest, 1);

        //var correct = (predictions == actual).sum().item<long>();
        //var accuracy = (double)correct / data.TestSetSize * 100.0;

        //Console.WriteLine($"Overall Accuracy: {accuracy:F2}%");
        //Console.WriteLine($"Correct: {correct}/{data.TestSetSize}");
    }
}

public class MCToolkit
{
    public static (torch.Tensor a, torch.Tensor b) Shuffle(torch.Tensor Z, torch.Tensor T)
    {
        var n = Z.shape[0];
        var indices = torch.randperm(n);
        return (Z.index(indices), T.index(indices));
    }

    public static torch.Tensor LoadImage(string filename, int IWidth, int IHeight)
    {
        Bitmap bm = new Bitmap(filename);
        if ((bm.Height != IHeight) || (bm.Width != IWidth))
            Console.WriteLine("Incorrect Image Size");

        float[] rgbPixels = new float[3 * bm.Width * bm.Height];
        int i = 0;
        for (int y = 0; y < bm.Height; y++)
        {
            for (int x = 0; x < bm.Width; x++)
            {
                Color rgb = bm.GetPixel(x, y);
                rgbPixels[i] = rgb.R / 255.0f;
                rgbPixels[i + bm.Height * bm.Width] = rgb.G / 255.0f;
                rgbPixels[i + 2 * bm.Height * bm.Width] = rgb.B / 255.0f;
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
                Color rgb = Color.FromArgb(
                    (int)(I[0][y][x].item<float>() * 255),
                    (int)(I[1][y][x].item<float>() * 255),
                    (int)(I[2][y][x].item<float>() * 255));
                bm.SetPixel(x, y, rgb);
            }
        }
        return bm;
    }
}

public class ImageDataLoaderxx
{
    public torch.Tensor z;
    public torch.Tensor t;
    public torch.Tensor ztest;
    public torch.Tensor ttest;
    public torch.Tensor zvalidation;
    public torch.Tensor tvalidation;

    public int IWidth = 64;
    public int IHeight = 64;
    public int NrClasses = 2;
    public int TrainSetSize = 0;
    public int TestSetSize = 0;
    public int ValidationSetSize = 0;

    public void LoadAndSave()
    {
        string[] classFolders = new string[]
        {
            @"C:\Users\th4te\OneDrive - Nelson Mandela University\A-NMU\4th Year\Semester 2\WRCV402\Assignment_6\human",
            @"C:\Users\th4te\OneDrive - Nelson Mandela University\A-NMU\4th Year\Semester 2\WRCV402\Assignment_6\non_human"
        };

        // First, collect and shuffle all files
        var allFiles = new System.Collections.Generic.List<(string path, int label)>();
        for (int i = 0; i < NrClasses; i++)
        {
            var files = Directory.GetFiles(classFolders[i]);
            foreach (var file in files)
            {
                allFiles.Add((file, i));
            }
        }

        // Shuffle all files together
        var rng = new Random();
        allFiles = allFiles.OrderBy(x => rng.Next()).ToList();

        int totalImages = allFiles.Count;
        TestSetSize = (int)(totalImages * 0.15);
        ValidationSetSize = (int)(totalImages * 0.15);
        TrainSetSize = totalImages - TestSetSize - ValidationSetSize;

        Console.WriteLine($"Total images: {totalImages}");
        Console.WriteLine($"Training: {TrainSetSize}, Validation: {ValidationSetSize}, Test: {TestSetSize}");

        // Initialize tensors
        z = torch.empty(TrainSetSize, 3, IHeight, IWidth);
        t = torch.zeros(TrainSetSize, NrClasses);
        ztest = torch.empty(TestSetSize, 3, IHeight, IWidth);
        ttest = torch.zeros(TestSetSize, NrClasses);
        zvalidation = torch.empty(ValidationSetSize, 3, IHeight, IWidth);
        tvalidation = torch.zeros(ValidationSetSize, NrClasses);

        // Load data in correct splits
        int index = 0;

        // Training data
        for (int i = 0; i < TrainSetSize; i++)
        {
            var (path, label) = allFiles[index++];
            z[i] = LoadAndTransform(path, IWidth, IHeight);
            t[i, label] = 1.0f;
        }

        // Validation data
        for (int i = 0; i < ValidationSetSize; i++)
        {
            var (path, label) = allFiles[index++];
            zvalidation[i] = LoadAndTransform(path, IWidth, IHeight);
            tvalidation[i, label] = 1.0f;
        }

        // Test data
        for (int i = 0; i < TestSetSize; i++)
        {
            var (path, label) = allFiles[index++];
            ztest[i] = LoadAndTransform(path, IWidth, IHeight);
            ttest[i, label] = 1.0f;
        }

        // Save data
        SaveData();
        Console.WriteLine("Data saved successfully");
    }

    public void LoadFromSaved()
    {
        z = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\ZData.pt");
        t = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\TData.pt");
        ztest = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\ZTData.pt");
        ttest = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\TTData.pt");
        zvalidation = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\ZVData.pt");
        tvalidation = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\TVData.pt");

        // Update sizes from loaded data
        TrainSetSize = (int)z.shape[0];
        TestSetSize = (int)ztest.shape[0];
        ValidationSetSize = (int)zvalidation.shape[0];
    }

    private void SaveData()
    {
        Directory.CreateDirectory(@"C:\Users\th4te\source\repos\AA66\Data");
        torch.save(z, @"C:\Users\th4te\source\repos\AA66\Data\ZData.pt");
        torch.save(t, @"C:\Users\th4te\source\repos\AA66\Data\TData.pt");
        torch.save(ztest, @"C:\Users\th4te\source\repos\AA66\Data\ZTData.pt");
        torch.save(ttest, @"C:\Users\th4te\source\repos\AA66\Data\TTData.pt");
        torch.save(zvalidation, @"C:\Users\th4te\source\repos\AA66\Data\ZVData.pt");
        torch.save(tvalidation, @"C:\Users\th4te\source\repos\AA66\Data\TVData.pt");
    }

    private static torch.Tensor LoadAndTransform(string path, int targetW, int targetH)
    {
        using (var bmp = new Bitmap(path))
        {
            int minDim = Math.Min(bmp.Width, bmp.Height);
            var cropRect = new Rectangle(
                (bmp.Width - minDim) / 2,
                (bmp.Height - minDim) / 2,
                minDim,
                minDim
            );

            using (var cropped = bmp.Clone(cropRect, bmp.PixelFormat))
            using (var resized = new Bitmap(targetW, targetH))
            using (var g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(cropped, 0, 0, targetW, targetH);

                float[] rgbPixels = new float[3 * targetW * targetH];
                int i = 0;

                for (int y = 0; y < targetH; y++)
                {
                    for (int x = 0; x < targetW; x++)
                    {
                        var pixel = resized.GetPixel(x, y);
                        rgbPixels[i] = pixel.R / 255f;
                        rgbPixels[i + targetW * targetH] = pixel.G / 255f;
                        rgbPixels[i + 2 * targetW * targetH] = pixel.B / 255f;
                        i++;
                    }
                }

                return torch.tensor(rgbPixels, dtype: ScalarType.Float32).reshape(3, targetH, targetW);
            }
        }
    }
}

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

        // Initialize weights
        InitializeWeights();
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

    private void InitializeWeights()
    {
        foreach (var (name, module) in this.named_modules())
        {
            if (module is Conv2d conv)
            {
                init.kaiming_normal_(conv.weight, mode: init.FanInOut.FanOut, nonlinearity: init.NonlinearityType.ReLU);
                if (conv.bias is not null)
                    init.constant_(conv.bias, 0);
            }
            else if (module is BatchNorm2d bn)
            {
                init.constant_(bn.weight, 1);
                init.constant_(bn.bias, 0);
            }
            else if (module is Linear linear)
            {
                init.normal_(linear.weight, 0, 0.01);
                init.constant_(linear.bias, 0);
            }
        }
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
        return x;
    }

    
}
