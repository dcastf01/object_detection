from enum import Enum
from typing import Union

# from config import CONFIG

from classification.model.architecture_available import ArchitectureAvailable

class ExperimentNames(Enum):
    TorchSqueezeNetDefaultLoss="torch_SqueezeNet_DefaultLoss"
    TorchSqueezeNetTripletLoss="torch_SqueezeNet_TripletLoss"
    TorchSqueezeNetDefaultLossAndTripletLoss="torch_SqueezeNet_DefaultLossAndTripletLoss"
    TorchtransFGDefaultLoss="torch_transFG_DefaultLoss"
    TorchtransFGTripletLoss="torch_transFG_TripletLoss"
    TorchtransFGDefaultLossAndTripletLoss="torch_transFG_DefaultLossAndTripletLoss"
    
class WrapperConfigModel():
    use_tripletLoss=False
    use_defaultLoss=False
    architecture_name=None

class ConfigArchitecture(WrapperConfigModel):
    
    def __init__(self,architecture_name:Union[str,Enum]):
        # super(ConfigArchitecture,self).__init__()
        self.architecture_name=self.check_architecture_name_on_list_and_return_architecture_name(architecture_name)
        self.pretrained=False
    def check_architecture_name_on_list_and_return_architecture_name(self,architecture_name):
        if architecture_name in ArchitectureAvailable.__members__:
            architecture_name=ArchitectureAvailable[architecture_name]
            return architecture_name
        elif  architecture_name in ArchitectureAvailable:
            
            return architecture_name      
        else:
            print(ArchitectureAvailable.__members__.keys())
            raise "you should pick a available model, here you have a list"


class ConfigModelDefaultLoss(WrapperConfigModel):
    
    def __init__(self):
        self.use_defaultLoss=True
        # ConfigModel.__init__(self,architecture_name=architecture_name,)
                                                    # default_Loss=default_Loss)
                                                    
class ConfigModelTripletLoss(WrapperConfigModel):
    
    def __init__(self):
        self.use_tripletLoss=True
        
    
class ConfigModelDefaultLossAndTripletLoss(ConfigModelTripletLoss,ConfigModelDefaultLoss):
    def __init__(self):
        ConfigModelTripletLoss.__init__(self)
        ConfigModelDefaultLoss.__init__(self)
        # super(ConfigModelTripletAndDefaultLoss,self).__init__(architecture_name=architecture_name)
    
class TorchSqueezenet(ConfigArchitecture ):
    def __init__(self):
        architecture_name=ArchitectureAvailable.torch_squeezenet
        ConfigArchitecture.__init__(self,architecture_name)
        
class TorchtransFG(ConfigArchitecture ):
    def __init__(self):
        architecture_name=ArchitectureAvailable.torch_transFG
        ConfigArchitecture.__init__(self,architecture_name)

class TorchSqueezeNetDefaultLoss(TorchSqueezenet,ConfigModelDefaultLoss):
    def __init__(self):
        TorchSqueezenet.__init__(self)
        ConfigModelDefaultLoss.__init__(self)

class TorchSqueezeNetTripletLoss(TorchSqueezenet,ConfigModelTripletLoss):
    def __init__(self):
        TorchSqueezenet.__init__(self)
        ConfigModelTripletLoss.__init__(self)
        
class TorchSqueezeNetDefaultLossAndTripletLoss(TorchSqueezenet,ConfigModelDefaultLossAndTripletLoss):
    def __init__(self):
        TorchSqueezenet.__init__(self)
        ConfigModelDefaultLossAndTripletLoss.__init__(self)

class TorchtransFGDefaultLoss(TorchtransFG,ConfigModelDefaultLoss):
    def __init__(self):
        TorchtransFG.__init__(self)
        ConfigModelDefaultLoss.__init__(self)

class TorchtransFGTripletLoss(TorchtransFG,ConfigModelTripletLoss):
    def __init__(self):
        TorchtransFG.__init__(self)
        ConfigModelTripletLoss.__init__(self)
        
class TorchtransFGDefaultLossAndTripletLoss(TorchtransFG,ConfigModelDefaultLossAndTripletLoss):
    def __init__(self):
        TorchtransFG.__init__(self)
        ConfigModelDefaultLossAndTripletLoss.__init__(self)



def get_config(name_experiment:ExperimentNames,model_pretrained:bool=False):
    CONFIGS_EXPERIMENTS={
        ExperimentNames.TorchSqueezeNetDefaultLoss:   TorchSqueezeNetDefaultLoss(),
        ExperimentNames.TorchSqueezeNetTripletLoss:   TorchSqueezeNetTripletLoss(),
        ExperimentNames.TorchSqueezeNetDefaultLossAndTripletLoss:   TorchSqueezeNetDefaultLossAndTripletLoss(),
        ExperimentNames.TorchtransFGDefaultLoss:   TorchtransFGDefaultLoss(),
        ExperimentNames.TorchtransFGTripletLoss:   TorchtransFGTripletLoss(),
        ExperimentNames.TorchtransFGDefaultLossAndTripletLoss:   TorchtransFGDefaultLossAndTripletLoss(),    
                        }
    config_experiment=CONFIGS_EXPERIMENTS[name_experiment]
    config_experiment.pretrained=model_pretrained
    return config_experiment



