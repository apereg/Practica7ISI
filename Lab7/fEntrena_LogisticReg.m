function theta = fEntrena_LogisticReg(X_train, Y_train, alpha)
% Esta funcion implementa el entrenamiento de un clasificador de regresion 
% logistica utilizando los datos de entrenamiento (X_train) y sus clases 
% (Y_train).  
%
% INPUT
%   - X_train: Matriz de dimension (m x n) con los datos de entrenamiento, 
%   donde m es el numero de patrones de entrenamiento y n es el numero de 
%   caracteristicas (la longitud del vector de caracteristicas que definen 
%   el patron.
%   - Y_train: Vector que contiene las clases de los patrones de 
%   entrenamiento. Su longitud sera m.
%   - alpha: Tasa de aprendizaje del algoritmo de descenso del gradiente
%
% OUTPUT
%   theta: Vector de longitud 1+n (es decir, un elemento mas que el numero 
%   de caracteristicas de cada patron). Contiene los parametros theta de 
%   la hipotesis estimada tras el entrenamiento.
%
% Author: Victor Gonzalez Castro
% Date: April 2020
% 

    % CONSTANTES
    % =================
    VERBOSE = true;
    max_iter = 100;
    % =================

    % Numero de patrones de entrenamiento.
    m = size(X_train,1);

    % Reservar espacio para los valores de la funcion de hipotesis para
    % cada patron de entrenamiento
    h_train = zeros(1, m);
    
    % Initialize
    theta = zeros(1, 1+size(X_train,2)); % Inicializa parametros de la hipotesis
    J = zeros(1, 1+max_iter); % Vector para almacenar el valor de la funcion de coste en cada iteracion

% *************************************************************************
% CALCULO DEL COSTE J PARA LAS THETAS INICIALES
    
    %a. Resultado intermedio: Calcular el error para cada elemento y sumarlo
    coste_total = 0;
    for i=1:m
        x_i = [1, X_train(i, :)]; % Coloca un 1 (el valor de x0) al principio de cada patron
        % Resultado de la hipotesis (salida esperada) para el patron i-esimo
        h_train(i) = fun_sigmoidal(theta, x_i);
        % Calculo del coste para el patron i-esimo y suma al coste de los patrones anteriores
        coste_patron_i = fCalculaCosteRegLog(Y_train(i), h_train(i));
        coste_total = coste_total + coste_patron_i;
    end
    
    % b. Calcula el coste
    J(1) = (-1/m)*coste_total;

% *************************************************************************
% DESCENSO DEL GRADIENTE PARA ACTUALIZACION DE THETA

    % Metodo iterativo que se realiza durante un numero maximo de
    % iteraciones
    for num_iter=1:max_iter
        
        % ***************************************************************
        % PASO 1:
        % Actualizar los valores de Theta
        theta_old = theta;
        % ====================== YOUR CODE HERE ======================
        % Resultado de la actualizacion de los parametros theta mediante 
        % descenso del gradiente
        sumatorio = 0;
        for i=1:m
             x_i = [1, X_train(i, :)];
            sumatorio = sumatorio + (h_train(i) - Y_train(i)) * x_i; 
        end
        theta= theta_old- alpha * 1/m* sumatorio;
        % ============================================================
        
        % ***************************************************************
        % PASO 2:
        % Calcular el valor de la funcion de hipotesis h (salida estimada) 
        % PARA CADA ELEMENTO DEL CONJUNTO DE ENTRENAMIENTO CON LOS NUEVOS 
        % THETA
        for i=1:m
            x_i = [1, X_train(i,:)]; % Coloca un 1 (el valor de x0) al principio de cada patron
            % Resultado de la hipotesis (salida esperada) para el patron i-esimo
            % ====================== YOUR CODE HERE ======================
            h_train(i) = fun_sigmoidal(theta, x_i);
            % ============================================================
        end

        % ***************************************************************
        % PASO 3. Calcular el coste en esta iteracion y almacenarlo en el
        % vector J
        
        %a. Resultado intermedio: Calcular el error PARA CADA ELEMENTO y 
        % sumar todos
        coste_total = 0;
        for i=1:m
            % ====================== YOUR CODE HERE ======================
            coste_total = coste_total + fCalculaCosteRegLog(Y_train(m), h_train(m));
            % ============================================================
        end
        
        % b. Calcular el valor de la funcion de coste J
        % ====================== YOUR CODE HERE ======================
        J(num_iter+1) = -(1/m)*coste_total;
        % ============================================================
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if VERBOSE
        % Dibuja el coste J en funcion del numero de iteracion
        figure;
        plot(0:num_iter, J, '-')
        title(['Cost function over the iterations with \alpha=', num2str(alpha)]);
        xlabel('Number of iterations');
        ylabel('Cost J');
    end

end