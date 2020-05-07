%------------------------------------
% CLASIFICADOR REGRESION LOGISTICA
% 
% Author: Victor Gonzalez Castro
% Date: April 2020
%------------------------------------

clear
close all

%% PASO INICIAL: CARGA CONJUNTO DE DATOS Y PARTICI?N TRAIN-TEST

load mammographic_data_norm.mat;
% X contiene los patrones de entrenamiento (dimension 5)
% Y contiene la clase del patron

% Numero de patrones (elementos) y de variables por cada patron en este dataset
[num_patrones, num_variables] = size(X);

% Parametro que indica el porcentaje de patrones que se utilizaran en 
% el conjunto de entrenamiento
p_train = 0.7;

% Particion de los datos en conjuntos de entrenamiento y test. 

num_patrones_train = round(p_train*num_patrones);
%num_patrones_test = num_patrones - num_patrones_train;

ind_permuta = randperm(num_patrones);

inds_train = ind_permuta(1:num_patrones_train);
inds_test = ind_permuta(num_patrones_train+1:end);

X_train = X(inds_train, :);
Y_train = Y(inds_train);

X_test= X(inds_test, :);
Y_test = Y(inds_test);

%% PASO 1: ENTRENAMIENTO DEL CLASIFICADOR Y CLASIFICACION DEL CONJUNTO DE TEST

% La funcion fClassify_LogisticReg implementa el clasificador de la regresion 
% logistica. Abrela y completa el codigo
alpha = 2;
umbral_decision = 0.5;

% ENTRENAMIENTO
theta = fEntrena_LogisticReg(X_train, Y_train, alpha);

% CLASIFICACION DEL CONJUNTO DE TEST
Y_test_hat = fClasifica_LogisticReg(X_test, theta);
Y_test_asig = Y_test_hat>=umbral_decision;

%% PASO 2: RENDIMIENTO DEL CLASIFICADOR: CALCULO DEl ACCURACY Y FSCORE

% Muestra matriz de confusion
figure;
plotconfusion(Y_test, Y_test_asig);

% Se obtienen los valores de la matriz de confusion usando expresiones
% logicas
TN = sum(Y_test_asig==0 & Y_test==0);
FN = sum(Y_test_asig==1 & Y_test==0);
TP = sum(Y_test_asig==1 & Y_test==1);
FP = sum(Y_test_asig==0 & Y_test==1);

% Error--> Error global -> Cambiado error a accuracy como decia en el video
% ====================== YOUR CODE HERE ======================
accuracy = (TP+TN)/(TP+FP+TN+FN);
% ============================================================
fprintf('\n******\nAccuracy = %1.4f%% (classification)\n', accuracy*100); 

% Sensitivity
% ====================== YOUR CODE HERE ======================
sensitivity = TP/(TP+FN);
% ============================================================

fprintf('\n******\nSensitivity = %1.4f (classification)\n', sensitivity);

% Specificity
% ====================== YOUR CODE HERE ======================
specificity = TN/(TN+FP);
% ============================================================

fprintf('\n******\nSpecificity = %1.4f (classification)\n', specificity);

% F-SCORE
% ====================== YOUR CODE HERE ======================
recall = TP/(TP+FN);
FScore = 2*(accuracy*recall/(accuracy+recall));
% ============================================================
fprintf('\n******\nFScore = %1.4f (classification)\n', FScore);
