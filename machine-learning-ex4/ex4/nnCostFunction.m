function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




%% =========== Part 1: Cost function  =============

% Turn y to a matrix
% I = eye(num_labels);
% Y = zeros(m, num_labels);
% for i = 1 : m
%   Y(i,:) = I(y(i), :);
% end
yd = eye(num_labels);
y = yd(y,:);

% Compute h
% Add ones to the X data matrix
X = [ones(m, 1) X];
a_2 = sigmoid( X * Theta1');
a_2 = [ones(m, 1) a_2];
h = sigmoid (a_2 * Theta2'); 

% Cost function wihtout regulation 
J = sum(sum((-y).*log(h)-(1-y).*log(1-h)))/m;

% Regulation term
reg_term = (0.5*lambda/m) * (sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% Cost function regularized 
J = J + reg_term;
% -------------------------------------------------------------

%% =========== Part 2: Backpropagation =============

delta_accum_1 = zeros(size(Theta1));
delta_accum_2 = zeros(size(Theta2));

% For each example
for t = 1:m
    
    % 1- Feedforward pass to get all the a's
    a_1 = X(t,:)';
    z_2 = Theta1*a_1;
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2]; 
    z_3 = Theta2*a_2;
    a_3 = sigmoid(z_3);
    
   % 2- get the delta_output for each unit
   %y_k = zeros(num_labels, 1);
   %y_k(y(t)) = 1;
   y_k = y(t,:)';
   delta_3 = a_3 - y_k;
     
   % 3- delta hidden layer
   delta_2 = Theta2' * delta_3 .* sigmoidGradient([1; z_2]);
   
   % 4- Accumulate the gradient
   	delta_accum_1 = delta_accum_1 + delta_2(2:end) * a_1';
	delta_accum_2 = delta_accum_2 + delta_3 * a_2';
    
end

% 5- unregularized gradient
Theta1_grad = delta_accum_1 / m;
Theta2_grad = delta_accum_2 / m;

%% =========== Part 3: Regularization =============


% Gradient Regularized
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda * Theta1(:,2:end) / m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda * Theta2(:,2:end) / m;
    
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
