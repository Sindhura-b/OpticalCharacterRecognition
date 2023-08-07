import deepdoctection as dd
from matplotlib import pyplot as plt
import os
os.environ['TESSDATA_PREFIX'] = '/home/vicky122/workspace/ga/DL/dl_project/OpticalCharacterRecognition/deepdoctection/'

analyzer = dd.get_dd_analyzer(reset_config_file=True)  # instantiate the built-in analyzer similar to the Hugging Face space demo
analyzer.get_pipeline_info()

image_path = "/home/vicky122/workspace/ga/DL/dl_project/OpticalCharacterRecognition/LARES/Good/T-40411/"

df = analyzer.analyze(path=image_path)  # setting up pipeline
df.reset_state()                 # Trigger some initialization

doc = iter(df)

for page in doc:

    name = page.file_name.split(".")[0]

    image = page.viz()
    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(image)
    plt.show()
    plt.imsave(f'{image_path}result_{name}.jpg', image)

    with open(f'{image_path}text_{name}.txt', 'w') as f:
        f.write(page.text)

    print(page.text)
    a=1