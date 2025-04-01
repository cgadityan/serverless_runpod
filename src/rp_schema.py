INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'garment': {
        'type': str,
        'required': True,
        'default': 'data/garment.jpg'
    },
    'mask': {
        'type': str,
        'required': True,
        'default': 'data/mask.png'
    },
    'model_img': {
        'type': str,
        'required': True,
        'default': 'data/model.jpg'
    },
    'output': {
        'type': str,
        'required' : False,
        'default': 'data/output.jpg'
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1224,
        # 'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1632,
        # 'constraints': lambda height: height in [128, 256, 384, 448, 512, 576, 640, 704, 768]
    },
    'prompt_strength': {
        'type': float,
        'required': False,
        'default': 0.8,
        'constraints': lambda prompt_strength: 0 <= prompt_strength <= 1
    },
    'num_outputs': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda num_outputs: 10 > num_outputs > 0
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50,
        'constraints': lambda num_inference_steps: 0 < num_inference_steps < 500
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 30,
        'constraints': lambda guidance_scale: 0 < guidance_scale < 40
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'K-LMS',
        'constraints': lambda scheduler: scheduler in ['DDIM', 'DDPM', 'DPM-M', 'DPM-S',  'EULER-A', 'EULER-D', 'HEUN', 'IPNDM', 'KDPM2-A', 'KDPM2-D', 'PNDM', 'K-LMS', 'KLMS']
    },
    'seed': {
        'type': int,
        'required': False,
        'default': 22
    },
    # 'nsfw': {
    #     'type': bool,
    #     'required': False,
    #     'default': False
    # },
    # 'lora': {
    #     'type': str,
    #     'required': False,
    #     'default': None
    # },
    # 'lora_scale': {
    #     'type': float,
    #     'required': False,
    #     'default': 1,
    #     'constraints': lambda lora_scale: 0 <= lora_scale <= 1
    # }
}
