# from flask import Flask
#
#
# # Elastic Beanstalk looks for an 'application' that is callable by default
# application = Flask(__name__)
#
#
# @application.route("/", methods=["GET", "POST"])
# def index():
#     return {"name":"Stephen Kamau1", "status":"OK"}
#
#
# # Run the application
# if __name__ == "__main__":
#     # Setting debug to True enables debug output. This line should be
#     # removed before deploying a production application.
#     application.debug = True
#     application.run(host="0.0.0.0")


# modules.
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import ConvertToMultiChannelBasedOnBratsClassesd, RandFlipd, CropForegroundd, SpatialPadd, EnsureTyped, NormalizeIntensityd, CropForegroundd
from monai.transforms import LoadImage, ScaleIntensity, EnsureChannelFirst, AddChannel, AsChannelFirst, ScaleIntensityd, RandRotate90d
from monai.transforms import LoadImaged, AddChanneld, ToTensord, Resized, Compose, EnsureChannelFirstd
from monai.transforms  import Invertd, SaveImaged, AsDiscreted, AsDiscrete
from monai.data import Dataset, list_data_collate, decollate_batch, pad_list_data_collate, CacheDataset, DataLoader

import torch
import imageio
#
import os
from flask import Flask, render_template, request, jsonify, url_for, redirect


# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = "cpu"
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to("cpu")


transformation_imgs = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(192, 192, 16), mode=('trilinear',), align_corners=False),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ]
)

model.load_state_dict(torch.load("./Best_model_Epoch_.pth", map_location=torch.device('cpu')))
print("Loaded the model")

from monai.inferers import sliding_window_inference
def create_inferences(img_zip_path, model, transformation_function):
    #apply transfomation
    test_img = transformation_function({"image":img_zip_path})['image'].reshape((-1,1,192,192,16))
    sample_pred = model(test_img)
    #sample_pred = sliding_window_inference(test_img.to(device), (240,240,240),1, model)

    print(sample_pred.shape)



    return sample_pred



application = Flask(__name__)



@application.get("/")
def index():
    return {"Status": "Neural Labs Africa okay"}


@application.route('/res', methods=['GET', 'POST'])
def create_upload_file():
    if request.method == 'POST':
        file = request.files['name']
        print("Already Done reading")

        if not file:
            return {"message": "No upload file Found"}
        else:
            print(f"Image FOund   {file.filename}")
            res = create_inferences(file.filename, model, transformation_imgs)
            res = torch.argmax(res, dim=1).detach().cpu()
            file_name = file.filename
            #save these segs
            sub_dir = f"{os.getcwd()}/{file_name.split('.')[0]}"
            # if not os.path.isdir(sub_dir):
            #
            #     os.mkdir(file_name.split(".")[0])
            # else:
            #     print("Path Found Alread")
            # for i in range(res.shape[-1]):
            #     full_slice_path = f"{sub_dir}/slice_{i}.png"
            #     #save.
            #     imageio.imwrite(full_slice_path, res[0,:,:,i])

            print("Done Inferencing")
            return {"filename": file.filename, "saved_dir":sub_dir}

    else:
        return jsonify({"message": "No upload file Found"})




@application.route("/<filename>", methods=["GET"])
def display(filename):
    print("Here")
    print(f"URL IS   {url_for('static', filename = f'./uploads/{filename}')}")
    return redirect(url_for('static', filename = f"./uploads/{filename}"))



# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run(host="0.0.0.0")
