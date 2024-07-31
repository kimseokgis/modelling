package jaro

import (
	"fmt"
	"testing"
)

func TestJaro(t *testing.T) {
	str1 := "cara membuat server"
	str2 := "membuat server dapat dilakukan dengan cara ....."
	score := jaroWinkler(str1, str2)
	fmt.Printf("Jaro-Winkler similarity between '%s' and '%s' is: %f\n", str1, str2, score)

}
