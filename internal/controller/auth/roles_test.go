package auth

import "testing"

func TestRoleAtLeast(t *testing.T) {
	roles := []string{RoleAdmin, RoleOperator, RoleMember, RoleViewer, "bogus"}
	// want[min] = roles that satisfy it
	want := map[string][]string{
		RoleAny:      {RoleAdmin, RoleOperator, RoleMember, RoleViewer},
		RoleViewer:   {RoleAdmin, RoleOperator, RoleViewer}, // members are not viewers: they see only their own data
		RoleMember:   {RoleAdmin, RoleOperator, RoleMember},
		RoleOperator: {RoleAdmin, RoleOperator},
		RoleAdmin:    {RoleAdmin},
	}
	for min, ok := range want {
		set := map[string]bool{}
		for _, r := range ok {
			set[r] = true
		}
		for _, r := range roles {
			if got := RoleAtLeast(r, min); got != set[r] {
				t.Errorf("RoleAtLeast(%q, %q) = %v, want %v", r, min, got, set[r])
			}
		}
	}
	if RoleAtLeast(RoleAdmin, "bogus") {
		t.Error("an unknown required role must never be satisfied")
	}
	for _, r := range []string{RoleAdmin, RoleOperator, RoleMember, RoleViewer} {
		if !ValidRole(r) {
			t.Errorf("%s should be valid", r)
		}
	}
	if ValidRole(RoleAny) {
		t.Error("'any' is a route requirement, not an account role")
	}
}
