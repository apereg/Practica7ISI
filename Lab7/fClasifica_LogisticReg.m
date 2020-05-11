function h = fClasifica_LogisticReg(X_test, theta)
% Esta funcion devuelve la probabilidad de cada patron del conjunto de test
% de pertenecer a la clase positiva utilizando regresion logistica.
%
% INPUT
%   - X_test: Matriz de dimension (m_t x n) con los datos de entrenamiento, 
%   donde m_t es el numero de patrones de entrenamiento y n es el numero de 
%   caracteristicas (la longitud del vector de caracteristicas que definen 
%   el patron.
%   - theta:  Vector de longitud n (es decir, el numero de caracteristicas de 
%   cada patron con los parametros theta de la hipotesis estimada tras el
%   entrenamiento.
%   
% OUTPUT
%	- h: Vector de longitud m_t (el numero de elementos en el conjunto de 
%   test) con las estimaciones realizadas para cada elemento de test por el
%   clasificador de regresion logistica (que corresponden a las 
%   probabilidades de que esos elementos correspondan a la clase positiva).
%
% Author: Victor Gonzalez Castro
% Date: April 2020
% 

    numElemTest = size(X_test,1);
    h = zeros(1, numElemTest);
    
    for i=1:numElemTest
        x_test_i = [1, X_test(i,:)];
        % ====================== YOUR CODE HERE ======================
        h(i) = fun_sigmoidal(theta, x_test_i); 
        % ============================================================
    end
end