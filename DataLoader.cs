using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static System.Reflection.Metadata.BlobBuilder;
using static TorchSharp.torch;
using System.Drawing.Imaging;

namespace CNNFunctions
{
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


}
