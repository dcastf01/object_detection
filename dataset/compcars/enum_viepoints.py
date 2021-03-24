import enum

class ViewPointEnum(enum.Enum):
    uncertain=-1
    front= 1
    rear=2
    side=3
    front_side=4
    rear_side=5
    
    def __str__(self):
        return self.name
def test():
    #get front with number
    print(ViewPointEnum(-1).name)
    
test()