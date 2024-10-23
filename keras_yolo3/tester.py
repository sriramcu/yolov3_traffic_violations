def inside(small,big) :
    # bottom-left and top-right corners of rectangle. 
    (x1, y1), (x2,  y2) = big
    for (x,y) in small:
        if (x > x1 and x < x2 and y > y1 and y < y2) : 
            return True
        
    return False
'''
person 0.94 [631, 42, 704, 236]
person 0.98 [576, 29, 693, 250]
motorbike 0.94 [590, 157, 668, 270]


'''


class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
  
# Returns true if two rectangles(l1, r1)  
# and (l2, r2) overlap 
def doOverlap(l1, r1, l2, r2): 
      
    # If one rectangle is on left side of other 
    if(l1.x >= r2.x or l2.x >= r1.x): 
        print("SIDE")
        return False
  
    # If one rectangle is above other 
    if(l1.y <= r2.y or l2.y <= r1.y): 
        print("Top")
        return False
  
    return True

if __name__ == '__main__':
    bike_boxes = [[590, 157, 668, 270]] #left,bottom ,right,top (ACTUAL order since vertical is reverse of cartesian)
    rider_boxes = [[631, 42, 704, 236],[576, 29, 693, 250]]
    pairs = []
    for bb in bike_boxes:
        for rb in rider_boxes:
            #if inside([(rb[0],rb[1]),(rb[2],rb[3])],[(bb[0],bb[1]),(bb[2],bb[3])]):#modify for function: bottom-left and top-right corners of rectangle. 
            #l1 is top left r1 is bottom right
            l1 = Point(bb[0],bb[3]) #590,157
            r1 = Point(bb[2],bb[1]) #668, 270
            l2 = Point(rb[0],rb[3]) #631, 42
            r2 = Point(rb[2],rb[1]) #704, 236
            
            if(doOverlap(l1, r1, l2, r2)):             
                pairs.append([rb,bb])
            else:
                print("Not inside")

    print(pairs)           
