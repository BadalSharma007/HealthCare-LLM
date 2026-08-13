export type NavItem = {
  href: string;
  label: string;
  description?: string;
};

export type NavSection = {
  title: string;
  items: NavItem[];
};

/** Dashboard navigation mapped to backend API tags */
export const navSections: NavSection[] = [
  {
    title: "Overview",
    items: [
      { href: "/", label: "Home", description: "Dashboard overview" },
    ],
  },
  {
    title: "Resume",
    items: [
      { href: "/resume", label: "Upload & fetch", description: "PDF upload / stored URL" },
      { href: "/resume/analyze", label: "Analyze", description: "Structured sections" },
      { href: "/resume/grading", label: "Grading", description: "Structure & metrics grade" },
      { href: "/resume/optimize", label: "Optimize", description: "Rewrite for a JD" },
      { href: "/resume/generate", label: "Generate CV", description: "Create formatted resume" },
    ],
  },
  {
    title: "Profiles",
    items: [
      { href: "/github", label: "GitHub", description: "Save & analyze profile" },
      { href: "/linkedin", label: "LinkedIn", description: "Scrape structured data" },
      { href: "/profile", label: "Job profile", description: "Aggregate profile + applied jobs" },
    ],
  },
  {
    title: "Matching",
    items: [
      { href: "/job-match", label: "Job match", description: "Score resume vs JD" },
      { href: "/similarity", label: "Similarity", description: "Work / project breakdown" },
    ],
  },
  {
    title: "Improvement",
    items: [
      { href: "/suggestions", label: "Suggestions", description: "Q&A bullet improvements" },
      { href: "/survey", label: "Experience survey", description: "Unwritten experience → content" },
    ],
  },
  {
    title: "Account",
    items: [
      { href: "/history", label: "History", description: "Action log" },
      { href: "/tokens", label: "Tokens", description: "Balance & transactions" },
      { href: "/settings", label: "Settings", description: "User details" },
    ],
  },
];
