# CS180 Project 5: Neural Radiance Field!

## Part 1: Fit a Neural Field to a 2D Image

### Method

#### Architecture

The model of my 2D neural field is a simple MLP with 4 hidden layer. The input dimension is $2\times(2\textbf L + 1)$ where $\textbf L$ is the level of positional encoding. The activation function between layers is `ReLU`. The output dimension is 3, representing the RGB value of the pixel. At the end of the MLP, I added a `Sigmoid` layer to constrain the network output be in the range of (0, 1).

In my implementation, I set the hidden dimension to be 256 and the number of layers to be 4. The learning rate is set to be 0.001. The batch size is set to be 16384. The number of epochs is set to be 300.

The positional encoding is a simple sine and cosine function. The input $x$ is first scaled to the range of (0, 1). Then, the positional encoding is defined as:

$$
P E(x)=\left\{x, \sin \left(2^0 \pi x\right), \cos \left(2^0 \pi x\right), \sin \left(2^1 \pi x\right), \cos \left(2^1 \pi x\right), \ldots, \sin \left(2^{L-1} \pi x\right), \cos \left(2^{L-1} \pi x\right)\right\}
$$

In my implementation, I set $L=10$.

#### Hyperparameters Tuning

I varied the level of PE and learning rate to find the best hyperparameters with the best PSNR.

First, I fixed the learning rate to be 0.01 and hidden dimension to be 256. Then, I trained the model with different level of PE: 6, 8, 10, 12, 14. The result is shown below:

![Positional Encoding](./images/2D/pe/pe.png)

From the figure, we can see that the PSNR decreases(get better) as the level increases before it reaches 10. After that, the PSNR does not change much. Therefore, I set the level of PE to be 10.

This makes sense because the input image is 1024x689. $2^{10}=1024$. Therefore, the input image can be encoded by the PE with level 10. When $L \leq 10$ the resolution of positional encoding is not enough to encode the position. When $L > 10$, $\forall i > 10, \sin \left(2^{i} \pi x\right)$ is redundant because it is the same as $\sin \left(2^{i-10} \pi x\right)$. Therefore, the PSNR does not change much.

Then, I fixed the level of PE to be 10 and hidden dimension to be 256. Then, I trained the model with different learning rate: 1e-1, 5e-2, 1e-2, 1e-3, 1e-4, 1e-5. The result is shown below:

![Learning Rate](./images/2D/lr/lr.png)

From the figure, we can see that when lr=1e-1 and 5e-2, the model cannot converge. When lr=1e-3, the model gets the best PSNR. When lr=1e-4 and 1e-5, the model converges but it takes more steps to reach the same PSNR as lr=1e-3. Therefore, I set the learning rate to be 1e-3.

#### Training PSNR

With the best hyperparameters given above, I trained the model for 300 epochs. The result is shown below:

![Training PSNR](./images/2D/best/best.png)

### Training Process Visualization

<div class="gallery">
    <figure>
        <img src="images/2D/version_1/0000.jpg" alt="Epoch 0">
        <figcaption>Epoch 0</figcaption>
    </figure>
    <figure>
        <img src="images/2D/version_1/0001.jpg" alt="Epoch 1">
        <figcaption>Epoch 1</figcaption>
    </figure>
    <figure>
        <img src="images/2D/version_1/0002.jpg" alt="Epoch 2">
        <figcaption>Epoch 2</figcaption>
    </figure>
    <figure>
        <img src="images/2D/version_1/0003.jpg" alt="Epoch 3">
        <figcaption>Epoch 3</figcaption>
    </figure>
    <figure>
        <img src="images/2D/version_1/0004.jpg" alt="Epoch 4">
        <figcaption>Epoch 4</figcaption>
    </figure>
    <figure>
        <img src="images/2D/version_1/0299.jpg" alt="Epoch 299">
        <figcaption>Epoch 299</figcaption>
    </figure>
    <style>
        .gallery {
            display: flex;
            justify-content: space-around;
            align-items: center;
        }
        .gallery img {
            width: 100px; /* Adjust as needed */
            height: auto;
        }
        .gallery figure {
            margin: 10px;
            text-align: center;
        }
    </style>
</div>

### Another Image Result

## Part 2: Fit a Neural Radiance Field from Multi-view Images

### Method

#### Create Rays from Cameras

#### Sampling

#### Neural Radiance Field

#### Volume Rendering

### Result

#### Rays and Samples

#### Training Process

#### PSNR Curve

#### Spherical Animation

### Bells and Whistles

#### Coarse-to-fine Sampling

#### Better PSNR

#### White Background

#### Depth Map Video

#### Nerfstudio
