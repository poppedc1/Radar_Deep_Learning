function x = input_select(type, t)
    if strcmp(type,'sin')
        %x = chirp(t,0,10,250,'quadratic',[],'concave');
        x = sin(t);
    elseif strcmp(type,'rect')
        x = rectpuls(t);

    elseif strcmp(type,'sinc')
        x = sinc(t);

    else
        print('not valid input type');
    end
end 