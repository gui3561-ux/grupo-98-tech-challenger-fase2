import { useState, type FormEvent } from "react";

interface Props {
  onSearch: (userId: string) => void;
  isLoading: boolean;
}

export function UserSearch({ onSearch, isLoading }: Props) {
  const [userId, setUserId] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (userId.trim()) {
      onSearch(userId.trim());
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-3">
      <input
        type="text"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
        placeholder="Enter User ID (e.g. 16332)"
        className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none"
      />
      <button
        type="submit"
        disabled={isLoading || !userId.trim()}
        className="rounded-lg bg-blue-600 px-6 py-2 font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {isLoading ? "Loading..." : "Get Recommendations"}
      </button>
    </form>
  );
}
