type PageHeaderProps = {
  title: string;
  description?: string;
  apiHint?: string;
};

export default function PageHeader({
  title,
  description,
  apiHint,
}: PageHeaderProps) {
  return (
    <div className="mb-6">
      <h2 className="text-2xl font-semibold tracking-tight">{title}</h2>
      {description ? (
        <p className="mt-1 max-w-2xl text-sm text-muted">{description}</p>
      ) : null}
      {apiHint ? (
        <p className="mt-2 font-mono text-xs text-slate-500">{apiHint}</p>
      ) : null}
    </div>
  );
}
