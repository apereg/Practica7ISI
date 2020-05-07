function cost_i = fCalculaCosteRegLog(y, y_hat)
% Calcula el coste de una salida del clasificador de regresion logistica
% (el resultado de aplicar la hipotesis h) y la clase real.
%
% INPUT
%   - y: Clase real.
%   - y_hat: Salida estimada calculada por la funcion de hipotesis de la
%   funcion de regresion logistica.
%
% OUTPUT
%   cost_i: Escalar con el valor del coste para la salida estimada y_hat de
%   entrada.
%
% Author: Victor Gonzalez Castro
% Date: April 2020
% 

    % ====================== YOUR CODE HERE ======================
    cost_i = y*log(y_hat) + (1-y)*log(1-y_hat);
    % ============================================================
    
end