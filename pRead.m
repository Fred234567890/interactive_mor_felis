function A=pRead(fName,varargin)
    cmplx=true;
    if nargin>1
       cmplx=varargin{1};
    end
    A=sparse(PetscBinaryRead(fName,'complex',cmplx,'indices','int64'));
end