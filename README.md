# Evos-Assignment-6
This is the official GitHub for Evolutionary Computing Group Assignment 6! 😋👍


1. Prepare Dataset
Organize images into separate folders like this...
Data/
  human/
  non_human/
Update the classFolders array in ImageDataLoader to match this file path.

2 First-Time Run (Preprocess and Save Images)
Comment out data.LoadFromSaved() in Program.Test1()
Uncomment data.LoadAndSave()
Run the program once to save to create the .pt files
.pt files will be created in path selected in dataloader
ZData.pt → training images
TData.pt → training labels
ZTData.pt → test images
TTData.pt → test labels

3. Subsequent Runs (Use Saved Tensors)
Comment data.LoadAndSave().
Uncomment data.LoadFromSaved().
This loads preprocessed tensors directly, saving time.

////Extra comments, I'm not sure where to put them.
{
.pt files improve performance/time by skipping repeated image preprocessing.
To add new images or change folders, run LoadAndSave() again to update tensors.
}

