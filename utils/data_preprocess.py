from transformers import AutoTokenizer
from torchvision import transforms
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from .conversation import conv_templates


def build_transform(
    input_size,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    # crop_pct = 224 / 256
    # size = int(input_size / crop_pct)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        # T.Resize(size, interpolation=InterpolationMode.BICUBIC),
        # T.CenterCrop(input_size),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return transform


DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<ImageContent>"
DEFAULT_IMG_START_TOKEN = "<img>"
DEFAULT_IMG_END_TOKEN = "</img>"
IGNORE_INDEX = -100


class RAHuskyCaptionCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        image_processor: transforms.Compose,
        num_queries: int,
        input_size: int,
        conv_template="multi_model",
        train_mode=True,
        max_length=192,
    ):
        self.train_mode = train_mode
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'right'
        self.image_processor = image_processor
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        image_query = DEFAULT_IMG_START_TOKEN + DEFAULT_IMAGE_TOKEN * num_queries + DEFAULT_IMG_END_TOKEN

        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], image_query + "\n{query}")
        conv.append_message(conv.roles[1], None)
        self.prompt_template = conv.get_prompt()
        self.max_length = max_length + num_queries + 6

        print('current max len:', self.max_length)
        print('current template:', self.prompt_template)

    def __call__(self, *args, **kwargs):
        if self.train_mode:
            return self._call_for_train(*args, **kwargs)

        return self._call_for_eval(*args, **kwargs)

    def _call_for_train(self, data_list):
        pixel_values = []
        input_ids = []
        attention_mask = []
        labels = []
        boxes = []
        boxes_mask = []
        for data in data_list:
            if 'pixel_values' in data:
                pixel_values.append(data['pixel_values'])
            else:
                pixel_values.append(self.image_processor(data['image']))

            curr_text_inputs = self.tokenizer(self.prompt_template.format(query=data['query'])).input_ids
            curr_labels = self.tokenizer(data['label'], add_special_tokens=False).input_ids
            curr_labels = curr_labels + [self.tokenizer.eos_token_id]

            input_ids.append(torch.tensor(curr_text_inputs + curr_labels, dtype=torch.long))
            attention_mask.append(torch.ones(len(curr_text_inputs) + len(curr_labels), dtype=torch.long))
            labels.append(torch.tensor([IGNORE_INDEX] * len(curr_text_inputs) + curr_labels, dtype=torch.long))

            box = data.get('bbox', (0, 0, data['image'].width, data['image'].height))
            box = (
                box[0] / data['image'].width * self.input_size[0],
                box[1] / data['image'].height * self.input_size[1],
                box[2] / data['image'].width * self.input_size[0],
                box[3] / data['image'].height * self.input_size[1],
            )
            # boxes.append(box)
            boxes.append(torch.tensor(box, dtype=torch.float32).unsqueeze(0))
            # boxes_mask.append('bbox' in data)
            boxes_mask.append(True)

        pixel_values = torch.stack(pixel_values, dim=0)
        # boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes_mask = torch.tensor(boxes_mask, dtype=torch.bool)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids[:, :self.max_length],
            'attention_mask': attention_mask[:, :self.max_length],
            'boxes': boxes,
            'boxes_mask': boxes_mask,
            'labels': labels[:, :self.max_length],
        }

    def _call_for_eval(self, data_list):
        pixel_values = []
        input_ids = []
        input_ids_end = []
        attention_mask = []
        attention_mask_end = []
        labels = []
        boxes = []
        boxes_mask = []
        for data in data_list:
            pixel_values.append(self.image_processor(data['image']))

            curr_text = self.prompt_template.format(query=data['query'])
            curr_text_inputs = self.tokenizer(curr_text).input_ids

            input_ids.append(torch.tensor(curr_text_inputs[:-1], dtype=torch.long))
            attention_mask.append(torch.ones(len(curr_text_inputs) - 1, dtype=torch.long))

            input_ids_end.append(curr_text_inputs[-1:])
            attention_mask_end.append([1])

            curr_labels = self.tokenizer(data['label'], add_special_tokens=False).input_ids
            curr_labels = curr_labels + [self.tokenizer.eos_token_id]
            labels.append(torch.tensor(curr_labels, dtype=torch.long))

            box = data.get('bbox', (0, 0, data['image'].width, data['image'].height))
            box = (
                box[0] / data['image'].width * self.input_size[0],
                box[1] / data['image'].height * self.input_size[1],
                box[2] / data['image'].width * self.input_size[0],
                box[3] / data['image'].height * self.input_size[1],
            )
            # boxes.append(box)
            boxes.append(torch.tensor(box, dtype=torch.float32).unsqueeze(0))
            boxes_mask.append('bbox' in data)

        pixel_values = torch.stack(pixel_values, dim=0)
        # boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes_mask = torch.tensor(boxes_mask, dtype=torch.bool)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        input_ids_end = torch.tensor(input_ids_end, dtype=torch.long)
        attention_mask_end = torch.tensor(attention_mask_end, dtype=torch.long)

        input_ids = torch.cat([input_ids, input_ids_end], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask_end], dim=1)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        meta_info = {
            'image_ids': torch.tensor([data['image_id'] for data in data_list], dtype=torch.long),
            'bboxes': torch.tensor([data['bbox'] for data in data_list], dtype=torch.float32),
            'labels': labels,
        }

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'boxes': boxes,
            'boxes_mask': boxes_mask,
            'labels': meta_info,
        }
