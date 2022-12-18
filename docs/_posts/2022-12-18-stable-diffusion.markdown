# Stable diffusion

## How Stable diffusion work

<https://scholar.harvard.edu/files/binxuw/files/stable_diffusion_a_tutorial.pdf>
<https://jalammar.github.io/illustrated-stable-diffusion/>
<https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion>

Stable Diffusion consists of three parts:

1. A **text encoder**, which turns your prompt into a latent vector.
2. A **diffusion model**, which repeatedly "denoises" a 64x64 latent image patch.
3. A **decoder**, which turns the final 64x64 latent patch into a higher-resolution 512x512 image.

![Stable diffusion architecture hight level](https://raw.githubusercontent.com/KhoaHoangViet/khoahoangviet.github.io/main/docs/_posts/assets/images/2022-12-18-stable_diffusion/sd-architecture-at-high-level.png)

First, your text prompt gets projected into a latent vector space by the text encoder, which is simply a pretrained, frozen language model

Then that prompt vector is concatenate to a randomly generated noise patch, which is repeatedly "denoised" by the decoder over a series of "steps"

Finally, the 64x64 latent image is sent through the decoder to properly render it in high resolution.

In order to do that we need:

- DDIM: Generate images by iteratively denoising pure noise
- **Conditioning**(what is this???????): The ability to control the generated visual contents via prompt keywords.

The ability to control the generated visual contents via prompt keywords. This is done via "conditioning", a classic deep learning technique which consists of concatenating to the noise patch a vector that represents a bit of text, then training the model on a dataset of {image: caption} pairs.

### Some concepts need to understand

#### VAE - Variable AutoEncoders

VAEs are a type of diffusion model that processes the encoding and decoding of data to prevent overfitting.

#### EMAs(Exponential Moving Average)

**Exponential moving average is a neural network training trick that sometimes improves the model accuracy**. Concretely, instead of using the optimized parameters from the final training iteration (parameter update step) as the final parameters for the model, the exponential moving average of the parameters over the course of all the training iterations are used.

<https://medium.com/analytics-vidhya/understanding-exponential-moving-averages-e3f020d9d13b#:~:text=%5B4%5D.-,EMA%20in%20Neural%20networks/Deep%20learning,-%3A%20In%20deep>

<https://www.reddit.com/r/MachineLearning/comments/ucflc2/d_understanding_the_use_of_ema_in_diffusion_models/>
It's an old trick to squeeze better performance out of SGD.
See e.g. Section 7.2 of the Adam paper (Temporal averaging): <https://arxiv.org/abs/1412.6980>

If I remember correctly it's only used at inference time yeah, the idea is that you have two sets of parameters, the set that is recently affected by what it recently saw during the training, and the set that is updated as an average over multiple iterations which supposedly have parameters which are more appropriate over the entire dataset.

This is because without EMA, models tend to overfit during the last iterations. With EMA the weights you use for inference are an average of all the weights you got during the last training iterations, which usually reduce this "last-iterations overfitting".

EMA isn't always easy to use in all applications because of this constraint, and doesn't always work. And if you don't know how it works and if there is a problem, you could have a high accuracy during training and a bad accuracy in test because the model used in EMA had a problem.

When I have a problem during inference and no problem during training, first thing I do is to remove EMA to be sure it's not the model using EMA which caused the issue. (Then I check batchnorm etc).

#### Sampling Methods

<https://stablediffusion.cdcruz.com/methods.html>
Please note that the differences stated are very minor and all sampling methods can produce good results, there is no drop in quality between methods but they do generate slight variations on an image even when using the same seed. there is no visual difference between methods and differences will only be based on how the internal calculations are made for rending images. 

More read: <https://jalammar.github.io/illustrated-stable-diffusion/>

#### DDIM (Denoising Diffusion Implicit Models)

<https://keras.io/examples/generative/ddim/>

Summary: Diffusion models are trained to denoise noisy images, and can generate images by iteratively denoising pure noise.

Latent diffusion

#### Kernel inception distance

Kernel Inception Distance (KID) is an image quality metric which was proposed as a replacement for the popular Frechet Inception Distance (FID)

#### U-Net

![U-Net](https://raw.githubusercontent.com/KhoaHoangViet/khoahoangviet.github.io/main/docs/_posts/assets/images/2022-12-18-stable_diffusion/unet-architecture.png)

U-Net is a popular semantic segmentation architecture, whose main idea is that it progressively downsamples and then upsamples its input image, and adds skip connections between layers having the same resolution. These help with gradient flow and avoid introducing a representation bottleneck, unlike usual autoencoders. Based on this, one can view diffusion models as denoising autoencoders without a bottleneck.


#### Super-resolution

It's possible to train a deep learning model to denoise an input image -- and thereby turn it into a higher-resolution version. The deep learning model doesn't do this by magically recovering the information that's missing from the noisy, low-resolution input -- rather, the model **uses its training data distribution to hallucinate the visual details** that would be most likely given the input

When you push this idea to the limit, you may start asking -- what if we just run such a model on pure noise? The model would then "denoise the noise" and start hallucinating a brand new image. By repeating the process multiple times, you can get turn a small patch of noise into an increasingly clear and high-resolution artificial picture.

Caption

## How to it is trained

### Fine tuning method

#### Analystic training model

<https://keras.io/examples/generative/random_walks_with_stable_diffusion>

#### Dreambooth

How Dreambooth work?

<https://github.com/nitrosocke/dreambooth-training-guide>
<https://www.reddit.com/r/StableDiffusion/comments/zi3g5x/new_15_dreambooth_model_analog_diffusion_link_in/>

#### Textual Diffusion

How Textual Diffusion work?
<https://www.reddit.com/r/sdforall/comments/yu43oj/textual_inversion_vs_dreambooth/>

##### Note from internet



#### Real fine tuning

##### Tutorial

<https://webbigdata.jp/ai/post-15346>

<https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning>

<https://gist.github.com/harubaru/f727cedacae336d1f7877c4bbe2196e1#training-process>

<https://github.com/huggingface/diffusers/tree/main/examples/text_to_image>

##### Create training data

Use CLIP to add prompt to image: <https://github.com/pharmapsychotic/clip-interrogator>
Underlying technology: The **CLIP Interrogator** is a prompt engineering tool that combines OpenAI's [CLIP](https://openai.com/blog/clip/) and Salesforce's [BLIP](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/) to optimize text prompts to match a given image. CLIP Interrogator uses [OpenCLIP](https://github.com/mlfoundations/open_clip) which supports many different pretrained CLIP models. For the best prompts for Stable Diffusion 1.X use ViT-L-14/openai for clip_model_name. For Stable Diffusion 2.0 use ViT-H-14/laion2b_s32b_b79k
[OpenAI Clip](https://openai.com/blog/clip/) and [Salesforce BLIP](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/)

In practice, it take about 50s for 1 image in NVIDIA T4

Something to try:

- Use regex to remove ", by {artist_name}" or change setting when generate prompt
- How to use multiple GPU to boost up prompt generation process??
- Add product type to prompt (whisky, soft drink, ...)
- Add product introduction(For example: [iyemon](https://products.suntory.co.jp/d/4901777317376/)) to text prompt after export some significant words

##### Create environment

Need A100 GPU (about 40GB in memory RAM), which supported by AWS but with high cost 8GPUs/32\$/1h and hard to change between instance type so we will use GCP instead (at 1GPUs/4\$/1h )

Prepare dataset and some more detail about trainig:
<https://huggingface.co/docs/diffusers/training/text2image>

<https://github.com/devilismyfriend/StableTuner>

#### Some other thing to be consider when training

##### XLA (Accelerated Linear Algebra)

<https://www.tensorflow.org/xla>

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes.

When a TensorFlow program is run, all of the operations are executed individually by the TensorFlow executor. Each TensorFlow operation has a precompiled GPU kernel implementation that the executor dispatches to.

XLA provides an alternative mode of running models: it compiles the TensorFlow graph into a sequence of computation kernels generated specifically for the given model. Because these kernels are unique to the model, they can exploit model-specific information for optimization

##### Speed up

<https://syncedreview.com/2022/11/09/almost-7x-cheaper-colossal-ais-open-source-solution-accelerates-aigc-at-a-low-cost-diffusion-pretraining-and-hardware-fine-tuning-can-be/>


## News to keep up

<https://www.reddit.com/r/StableDiffusion/comments/znli8v/where_can_i_keep_up_with_stable_diffusion_that/>

<https://rentry.org/sdupdates3>

<https://www.reddit.com/r/StableDiffusion/comments/ycm5qe/my_posts_of_stable_diffusion_links_that_i/?sort=new>

## Things need to be consider (really really important)

<https://jamesblaha.medium.com/the-problem-isnt-ai-it-s-requiring-us-to-work-to-live-3cb4a4b468e9>
<https://www.reddit.com/r/StableDiffusion/comments/zlhin3/the_problem_isnt_ai_its_requiring_us_to_work_to/>
