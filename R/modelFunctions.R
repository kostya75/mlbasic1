
# 3.2 initialize parameters deep
#'@importFrom stats rnorm
initialize_parameters_deep<-function(layer_dims){
  set.seed(12) #comment out
  parameters<-list()
  L<-length(layer_dims)
  for(l in 2:L){
    #parameters[[sprintf("W%s",l-1)]]<-matrix(rnorm(layer_dims[l-1]*layer_dims[l])*.01,ncol=layer_dims[l-1])
    parameters[[sprintf("W%s",l-1)]]<-matrix(rnorm(layer_dims[l-1]*layer_dims[l]),ncol=layer_dims[l-1])/sqrt(layer_dims[l-1])
    parameters[[sprintf("b%s",l-1)]]<-ones_zeros(0,c(layer_dims[l],1))
  }
  return(parameters)
}



# 4.1 Linear forward

linear_forward<-function(A, W, b){

  # Arguments:
  # A -- activations from previous layer (or input data): (size of previous layer, number of examples)
  # W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
  # b -- bias vector, numpy array of shape (size of the current layer, 1)
  #
  # Returns:
  # Z -- the input of the activation function, also called pre-activation parameter
  # cache -- a list  containing "A", "W" and "b" ; stored for computing the backward pass efficiently
  #
  b_broadcast<-dim(A)[2]
  #b_calc used to broadcast
  b_calc<-b[,rep(1,times=b_broadcast)]
  Z<-W%*%A+b_calc
  stopifnot(dim(Z)==c(dim(W)[1],dim(A)[2]))
  return(list(Z=Z,cache=list(A=A,W=W,b=b)))
}



# 4.2 linear activation forward
linear_activation_forward<-function(A_prev, W, b, activation){
  if(activation=="sigmoid"){

    Z_linear_cache<-linear_forward(A_prev, W, b)
    Z<-Z_linear_cache[["Z"]]
    linear_cache<-Z_linear_cache[["cache"]]

    A_activation_cache<-sigmoid(Z)
    A<-A_activation_cache[["A"]]
    activation_cache<-A_activation_cache[["cache"]]
  }
  if (activation=="relu"){
    Z_linear_cache<-linear_forward(A_prev, W, b)
    Z<-Z_linear_cache[["Z"]]
    linear_cache<-Z_linear_cache[["cache"]]

    A_activation_cache<-relu(Z)
    A<-A_activation_cache[["A"]]
    activation_cache<-A_activation_cache[["cache"]]
  }
  stopifnot(dim(A)==c(dim(W)[1],dim(A_prev)[2]))
  cache<-list(linear_cache=linear_cache,activation_cache=activation_cache)
  return(list(A=A,cache=cache))
}



L_model_forward<-function(X, parameters){
  caches<-list()
  A<-X
  L<-as.integer(length(parameters)/2)

  for (l in 1:(L-1)){
    A_prev<-A
    A_cache<-linear_activation_forward(A_prev, parameters[[sprintf("W%s",l)]], parameters[[sprintf("b%s",l)]], activation="relu")
    A<-A_cache$A
    caches[[l]]<-A_cache$cache
  }
  AL_cache = linear_activation_forward(A, parameters[[sprintf("W%s",L)]], parameters[[sprintf("b%s",L)]], activation="sigmoid")
  AL<-AL_cache$A
  caches[[L]]<-AL_cache$cache
  stopifnot(dim(AL)==c(1,dim(X)[2]))
  return(list(AL=AL,caches=caches))
}


# 5.0 compute cost

compute_cost<-function(AL, Y){
  m=dim(Y)[2]
  cost<-unname((-1/m)*(log(AL)%*%t(Y)+log(1-AL)%*%(1-t(Y)))[1,1])
  stopifnot(is.null(dim(cost)))
  return(cost)
}



################## backward #####################

# 6.1 linear backward

linear_backward<-function(dZ,cache){
  A_prev<-cache$A
  W<-cache$W
  b<-cache$b
  m<-dim(A_prev)[2]

  ######### 3 equations
  dW<-dZ%*%t(A_prev)/m
  # broadcast only when using b in a formula
  #db_broadcast<-dim(b)[2]
  db<-matrix(rowSums(dZ),ncol=1,byrow=T)/m
  #db<-db[,rep(1,times=db_broadcast)]
  dA_prev<-t(W)%*%dZ
  ######### 3 equations

  stopifnot(dim(A_prev)==dim(dA_prev))
  stopifnot(dim(W)==dim(dW))
  stopifnot(dim(b)[1]==dim(db)[1])

  return(list(dA_prev=dA_prev,dW=dW,db=db))
}



# 6.2 linear activation backward

linear_activation_backward<-function(dA,cache,activation){
  linear_cache<-cache$linear_cache
  activation_cache<-cache$activation_cache
  if(activation=="relu"){
    dZ<-relu_backward(dA, activation_cache)
    dA_prev_dW_db<-linear_backward(dZ, linear_cache)
    dA_prev<-dA_prev_dW_db$dA_prev
    dW<-dA_prev_dW_db$dW
    db<-dA_prev_dW_db$db
  }
  if(activation=="sigmoid"){
    dZ<-sigmoid_backward(dA, activation_cache)
    dA_prev_dW_db<-linear_backward(dZ, linear_cache)
    dA_prev<-dA_prev_dW_db$dA_prev
    dW<-dA_prev_dW_db$dW
    db<-dA_prev_dW_db$db
  }
  return(list(dA_prev=dA_prev, dW=dW, db=db))
}


# 6.3 L-Model backward
#tt[[1]]

L_model_backward<-function(AL,Y,caches){
  grads<-list()
  L<-length(caches)
  m<-dim(AL)[2]
  dAL<-(-1)*(Y/AL-((1-Y)/(1-AL)))
  current_cache<-caches[[L]]

  grads[[L]]<-linear_activation_backward(dAL,current_cache,"sigmoid")
  names(grads[[L]])<-c(sprintf("dA%s",L-1),sprintf("dW%s",L),sprintf("db%s",L))

  for(l in rev(1:(L-1))){
    current_cache<-caches[[l]]
    grads[[l]]<-linear_activation_backward(grads[[l+1]][[sprintf("dA%s",l)]],current_cache,"relu")
    names(grads[[l]])<-c(sprintf("dA%s",l-1),sprintf("dW%s",l),sprintf("db%s",l))
  }
  return(grads)
}



# 6.4 Update parameters

update_parameters<-function(parameters, grads, learning_rate){
  L<-as.integer(length(parameters)/2)
  for(l in 1:L){
    parameters[[sprintf("W%s",l)]] = parameters[[sprintf("W%s",l)]] - learning_rate*grads[[l]][[sprintf("dW%s",l)]]
    parameters[[sprintf("b%s",l)]] = parameters[[sprintf("b%s",l)]] - learning_rate*grads[[l]][[sprintf("db%s",l)]]
  }
  return(parameters)
}

# final model

# 4. Model: L-layer

#'Main function for multilayer NN estimation. Binary choice as output layer. Hidden layers use 'relu', output layer is 'sigmoid'
#'
#'@param X matrix of features. Each column represents one training example
#'@param Y matrix (1 x m) of true responses
#'@param layers_dims structure of the model
#'@param learning_rate spcecify the learning rate for gradient descent
#'@param num_iterations number of iterations for gradient descent
#'@param print_cost boolean, whether or not to print cost
#'
#'@examples
#'\dontrun{
#'first_layer<-nrow(X)
#'model_dim<-c(first_layer,,l2,l3,...,1)
#'parameters<-L_layer_model(X, Y, layers_dims=model_dim)
#'}
#'@export
L_layer_model<-function(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=F){
  set.seed(1)
  cost<-NULL
  parameters<-initialize_parameters_deep(layers_dims)
  for(i in 1:num_iterations){

    # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
    #AL, caches = L_model_forward(X, parameters)
    AL_caches<-L_model_forward(X, parameters)
    AL<-AL_caches[["AL"]]
    caches<-AL_caches[["caches"]]

    # Compute cost.
    cost<-compute_cost(AL, Y)

    # Backward propagation.
    grads<-L_model_backward(AL, Y, caches)

    # Update parameters.
    parameters<-update_parameters(parameters, grads, learning_rate)
    #print(i)
    # print cost
    if(print_cost & (i %% 100) ==0){
      print(cost)
    }
  }
  return(parameters)
}


# predict
#'Function creates prediction based on the data and patameter estimates
#'
#'@param X a matrix containing each training example as column
#'@param parameters a matrix of weights
#'
#'@export
predict_nn_L_layer<-function(X, parameters){
  AL<-L_model_forward(X, parameters)[["AL"]]
}
