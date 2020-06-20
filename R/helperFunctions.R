# relu_backward calculated drelu x dA. it returns dZ not drelu
relu_backward<-function(dA,cache,leaky=0){
  Z<-cache
  temp_compare<-structure(vapply(Z,
                                 function(x) if(x>=0) 1 else leaky
                                 ,numeric(1)),
                          dim=dim(Z))
  dZ<-dA*temp_compare
}

# dA<-matrix(c(1,2,-3,4,2,3,4,5),nrow=2,byrow=T)
# Z<-matrix(c(3,5,-1,-8,4,-4,4,-4),nrow=2,byrow=T)
# (relu_backward(dA,Z))

# relu forward
relu<-function(Z,leaky=0){
  A<-structure(pmax(0,Z),dim=dim(Z))
  A[Z<0]<-Z[Z<0]*leaky
  #check_Z(Z)
  #A[Z<0]<-0
  return(list(A=A,cache=Z))
}

# debugging
# check_Z<-function(Z){
#   if(is.na(Z)){
#     browser()
#     stop("Z is na")
#   }
# }

#Z<-matrix(c(1,2,-3,4,3,5,-1,-8),nrow=2,byrow=T)

# sigmoid with cache
sigmoid<-function(Z){
  A<-1/(1+exp(-Z))
  return(list(A=A,cache=Z))
}


# sigmoid backward: gradient x dA
sigmoid_backward<-function(dA, cache){
  Z<-cache
  s<-1/(1+exp(-Z))
  dZ<-dA*s*(1-s)
}

#(sigmoid_backward(dA,Z))

# pass matrix size to create empty matrix of same size with a value
ones_zeros<-function(value,size){
  if(length(size)!=2  | !is.numeric(size)) stop("Size should have length of 2 and be numeric")
  matrix(value,nrow=size[1],ncol=size[2])
}


