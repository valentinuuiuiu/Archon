-- Create archon_agents table for storing dynamic agent definitions
CREATE TABLE IF NOT EXISTS archon_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(255) NOT NULL,
    description TEXT,
    system_prompt TEXT NOT NULL,
    model VARCHAR(255) DEFAULT 'openai:gpt-4o',
    tools JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255) DEFAULT 'system'
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_archon_agents_name ON archon_agents(name);

-- Add trigger to automatically update updated_at timestamp
-- Note: update_updated_at_column function should already exist from complete_setup.sql
CREATE TRIGGER update_archon_agents_updated_at
    BEFORE UPDATE ON archon_agents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security
ALTER TABLE archon_agents ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
-- Service role has full access
CREATE POLICY "Allow service role full access to archon_agents" ON archon_agents
    FOR ALL USING (auth.role() = 'service_role');

-- Authenticated users can read and write (since we want users to create agents)
CREATE POLICY "Allow authenticated users to read and update archon_agents" ON archon_agents
    FOR ALL TO authenticated
    USING (true);

-- Record migration
INSERT INTO archon_migrations (version, migration_name)
VALUES ('0.2.0', '009_create_agents_table')
ON CONFLICT (version, migration_name) DO NOTHING;
