% PID self Tunning For Nonlinear Systems

clc;
clear;
close all;
alpha=0.01;
itNN=180;
dt=0.01;
n=itNN-1;
Neuron=10;    %Number of neurons in hidden layers
NumInt=7;    %Number of inputs
NumInt3=3;   %Number of last Hidden Layer
P=zeros(1,itNN);
I=zeros(1,itNN);
D=zeros(1,itNN);
f=zeros(1,itNN);
Vdc=zeros(1,itNN);

Theta1=zeros(1,itNN);

% input 1
ref=randi([60 60],1,60);
ref2=randi([20 20],1,60);
ref33=randi([40 40],1,60);
ref3=[ref ref2 ref33];

%reference Model
refe2=randi([60 60],1,60);
refe22=randi([20 20],1,60);
refe33=randi([40 40],1,60);
reference=[refe2 refe22 refe33];

% input 2
% ref=randi([30 30],1,60);
% ref2=randi([20 20],1,60);
% ref33=randi([50 50],1,60);
% ref3=[ref ref2 ref33];

%Reference Model 2
% tt=linspace(-10,10,180);
% reference=sin(tt);

%% Initialize DRNN Controller

        
% Weights
W_w.w1=[];        % Weights From inputs to  Hidden layer1
W_w.w2=[];        % Weights From Hidden layer1 to Hidden layer2
W_w.w3=[];        % Weights From Hidden layer2 to Hidden layer3
W_w.w4=[];        % Weights From Hidden layer3 to output node
W_w.u=[];


Ww=repmat(W_w,itNN ,1);

for k=1:itNN
    
  Ww(k).w1=unifrnd(0,1,NumInt,Neuron);
  Ww(k).w2=unifrnd(0,1,Neuron,Neuron);
  Ww(k).w3=unifrnd(0,1,NumInt3,Neuron);
  Ww(k).w4=unifrnd(0,1,1,NumInt3);
  
end
u=zeros(1,itNN);
E=zeros(1,itNN);                              %Error between Theta and reference
E1=zeros(1,itNN);                             % observing and reduce Error
E2=zeros(1,itNN);                            

for it=1:3000
for i=2:n

  Input_of_Hidden_layer1 =Ww(i).w1'*[ref3(i) E1(i) E1(i-1) Theta1(i) Theta1(i-1) u(i-1) u(i)]'+1;


    output_ofHidden_layer1 = 1./(1+exp(-Input_of_Hidden_layer1));
     
     Input_of_Hidden_layer2 = (Ww(i).w2*output_ofHidden_layer1)+1;
     
     output_ofHidden_layer2 =  1./(1+exp(-Input_of_Hidden_layer2));
     
     Input_of_Hidden_layer3 = (Ww(i).w3*output_ofHidden_layer2);
     
     output_ofHidden_layer3 =  1./(1+exp(-Input_of_Hidden_layer3));
     
     Input_of_output_Node =   (Ww(i).w4*output_ofHidden_layer3)+1;
     
     u(i)=Input_of_output_Node; % out put of output layer(linear)

    
    Theta1(i+1)=0.9*Theta1(i)-0.001*Theta1(i-1)^2+u(i)+sin(u(i-1));
    E1(i+1)=ref3(i+1)-Theta1(i+1);
   
    P(i+1)=Ww(i).w4(1,1)*E1(i+1);
 	I(i+1)=Ww(i).w4(1,2)*(I(i)+dt*E1(i)+0.5*dt*(E1(i+1)-E1(i)));
 	D(i+1)=Ww(i).w4(1,3)*(E1(i+1)-E1(i))/dt;  
    u(i+1)=P(i+1)+I(i+1)+D(i+1);
            if u(i)>10
                u(i)=10;
            elseif u(i)<0
                u(i)=0;
            end
                
              
% Back Propagation

   E2(i)=(reference(i+1)-Theta1(i+1));
   E(i)=(reference(i)-u(i));

   error_of_hidden_layer3=Ww(i).w4'*E2(i);
   
   Delta3=error_of_hidden_layer3.*(Input_of_Hidden_layer3>0);
    
   error_of_hidden_layer2=Ww(i).w3'*Delta3;
   Delta2=(Input_of_Hidden_layer2>0).*error_of_hidden_layer2;
    
   error_of_hidden_layer1=Ww(i).w2'*Delta2;
   Delta1=(Input_of_Hidden_layer1>0).*error_of_hidden_layer1;
    
   adjustment_of_W4= alpha*E2(i) * output_ofHidden_layer3';
   adjustment_of_W3= alpha*Delta3 * output_ofHidden_layer2';
   adjustment_of_W2= alpha*Delta2* output_ofHidden_layer1';
   adjustment_of_W1= alpha*Delta1*[ref3(i) E1(i) E1(i-1) Theta1(i) Theta1(i-1) u(i-1) u(i)];
   adjustment_of_W1=adjustment_of_W1';
    %uptodata Weights
    Ww(i).w1=Ww(i).w1+adjustment_of_W1;
    Ww(i).w2=Ww(i).w2+adjustment_of_W2;
    Ww(i).w3=Ww(i).w3+adjustment_of_W3;
    Ww(i).w4=Ww(i).w4+adjustment_of_W4;


end
end

%% Results

 % Result 
 figure;
  plot(reference,'LineWidth',2)   
  hold on;
  plot(Theta1,'r','LineWidth',1.2)
  plot(u)
  legend('reference','system output','Control signal')
  title('PID NN Controler')
  hold off
