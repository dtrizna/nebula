# instantiate the model for the downstream task with the same number of classes as the pre-trained model
model = PreTrainedModel(num_classes=5)

# modify the output layer to have the correct number of classes for the downstream task
model.out = nn.Linear(30, 10)

# load the pre-trained weights, except for the output layer
pretrained_dict = torch.load('pretrained_model.pt')
model_dict = model.state_dict()

# filter out the output layer weights from the pre-trained weights
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != 'out.weight' and k != 'out.bias'}

# update the model's weights
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
