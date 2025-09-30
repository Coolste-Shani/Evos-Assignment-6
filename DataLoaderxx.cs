
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
    public class ImageDataLoaderxx
    {
        public torch.Tensor z;          //Training set inputs
        public torch.Tensor t;          //Training set expected outputs
        public torch.Tensor ztest;      //Test set inputs
        public torch.Tensor ttest;      //Test set expected outputs

        public int IWidth = 224;
        public int IHeight = 224;
        public int NrClasses = 2;
        public int TrainSetSize = 40;
        public int TestSetSize = 10;

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

                // Training set
                for (int p = 0; p < NrTrainImages; p++)
                {
                    z[zindex] = LoadAndTransform(files[p], IWidth, IHeight);

                    var ttemp = torch.zeros(NrClasses);
                    ttemp[i] = 1.0f;
                    t[zindex] = ttemp;

                    zindex++;
                }

                // Test set
                for (int p = NrTrainImages; p < files.Length; p++)
                {
                    ztest[tindex] = LoadAndTransform(files[p], IWidth, IHeight);

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
            z = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\ZData.pt");
            t = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\TData.pt");
            ztest = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\ZTData.pt");
            ttest = torch.load(@"C:\Users\th4te\source\repos\AA66\Data\TTData.pt");
        }

        // 🔹 New helper: resize + center crop + normalize to tensor
        private static torch.Tensor LoadAndTransform(string path, int targetW, int targetH)
        {
            using (var bmp = new Bitmap(path))
            {
                // Center crop to square
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
}
