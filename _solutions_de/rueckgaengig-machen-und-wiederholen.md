---
title: Rückgängig machen und Wiederholen
description: Nutzern erlauben, Aktionen rückgängig zu machen und erneut
  anzuwenden, zur Fehlerbehebung und zum Erkunden.
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/undo-and-redo/
problems:
- user-frustration
- poor-user-experience-ux-design
- user-trust-erosion
- increased-error-rates
- fear-of-change
- negative-user-feedback
- customer-dissatisfaction
layout: solution
lang: de
en_slug: undo-and-redo
related_solutions:
- slug: confirmation-dialogs
  similarity: 0.75
- slug: auto-save
  similarity: 0.75
- slug: intuitive-navigation
  similarity: 0.75
- slug: search-function
  similarity: 0.7
- slug: understandable-error-messages
  similarity: 0.7
- slug: plain-language
  similarity: 0.7
---

## Description

Rückgängig machen und Wiederholen erlauben einem Nutzer, eine Aktion rückgängig zu machen und erneut anzuwenden, und verwandeln, was sonst eine dauerhafte, irreversible Operation wäre, in etwas, das sicher zu versuchen und leicht zurückzunehmen ist. Legacy-Systeme, die dies nie implementierten, zwingen Nutzer in eine defensive Haltung — sie kopieren ganze Datensätze oder Seiten in Backup-Ordner, bevor sie irgendeine Bearbeitung vornehmen, bewegen sich langsam und ängstlich, weil jede Aktion die sein könnte, die sie nicht rückgängig machen können — was eine direkte Steuer sowohl auf Geschwindigkeit als auch auf die Bereitschaft ist, die tatsächlichen Fähigkeiten des Systems zu erkunden. Ein vollständiger Aktionshistorie-Stack ist die vollständigste Version davon, aber ein einfacherer, zeitlich begrenzter "Rückgängig"-Link bei einer Erfolgsbestätigung handhabt den häufigsten Fall zu weit geringeren Implementierungskosten, und für alles, was die Datenbank betrifft, ist der Bau von Rückgängig-Funktionalität über Soft Deletes oder einen Audit-Trail weit handhabbarer als der Versuch, eine Legacy-Transaktion direkt umzukehren.

## How to Apply ◆

> Legacy-Systeme unterstützen selten Rückgängig-Funktionalität, was jede Aktion potenziell irreversibel macht und Nutzer vorsichtig, langsam und ängstlich werden lässt. Das Hinzufügen von Rückgängig- und Wiederherstellen-Fähigkeit ermöglicht zuversichtliches Erkunden und schnelle Fehlerbehebung.

- Implementieren Sie einen Aktionshistorie-Stack, der Nutzermodifikationen protokolliert und erlaubt, sie rückgängig zu machen. Beginnen Sie mit den häufigsten und folgenreichsten Aktionen, statt zu versuchen, alles auf einmal rückgängig machbar zu machen.
- Unterstützen Sie mehrstufiges Rückgängig-Machen, das Nutzern erlaubt, eine Sequenz von Aktionen umzukehren, nicht nur die letzte. Zeigen Sie die Rückgängig-Historie an, sodass Nutzer sehen können, was sie rückgängig machen.
- Implementieren Sie Wiederholen zusammen mit Rückgängig, sodass Nutzer, die versehentlich zu weit rückgängig machen, ihre Änderungen erneut anwenden können, ohne sie neu einzugeben.
- Bieten Sie eine Rückgängig-Option in Erfolgsbestätigungen an, wie "Element gelöscht. [Rückgängig]" mit einem zeitlich begrenzten Fenster für die Umkehrung. Dies ist einfacher zu implementieren als vollständige Rückgängig-Historie und handhabt den häufigsten Anwendungsfall.
- Implementieren Sie für Datenbankoperationen Rückgängig-Machen über Soft Deletes, Audit-Trails oder Event Sourcing, statt zu versuchen, Datenbanktransaktionen umzukehren, was in Legacy-Systemen komplex und brüchig ist.
- Kommunizieren Sie die Rückgängig-Fähigkeit klar durch Tastaturkürzel-Unterstützung (Strg+Z / Cmd+Z) und sichtbare Rückgängig-Schaltflächen, sodass Nutzer wissen, dass das Sicherheitsnetz existiert.

## Tradeoffs ⇄

> Rückgängig-Fähigkeit verändert das Nutzerverhalten grundlegend, indem sie die Angst vor Fehlern beseitigt, aber die Implementierung in einem Legacy-System kann architektonisch herausfordernd sein.

**Vorteile:**

- Beseitigt die Angst vor Fehlern, die Nutzer davon abhält, Features zu erkunden oder Änderungen in Legacy-Systemen vorzunehmen.
- Reduziert die Konsequenzen versehentlicher Aktionen dramatisch, was den Bedarf an Datenwiederherstellung durch Administratoren verringert.
- Ermöglicht schnelles Experimentieren, weil Nutzer Dinge ausprobieren und leicht rückgängig machen können, wenn das Ergebnis nicht dem entspricht, was sie erwarteten.
- Baut Nutzervertrauen auf, indem ein sichtbares Sicherheitsnetz bereitgestellt wird, das demonstriert, dass das System die Fähigkeit der Nutzer respektiert, ihre Meinung zu ändern.

**Kosten und Risiken:**

- Die Implementierung von Rückgängig-Machen in einem Legacy-System mit direkten Datenbankschreibvorgängen und ohne Audit-Trail erfordert architektonische Änderungen, um umkehrbare Aktionen zu protokollieren.
- Manche Operationen sind inhärent schwierig oder unmöglich rückgängig zu machen, wie das Senden von E-Mails, das Auslösen externer API-Aufrufe oder das Starten physischer Prozesse. Klare Kommunikation darüber, welche Aktionen rückgängig machbar sind, ist essenziell.
- Die Rückgängig-Historie verbraucht Speicherplatz und muss begrenzt werden, um unbegrenztes Wachstum zu verhindern, besonders in hochvolumigen Systemen.
- Mehrbenutzerumgebungen schaffen Komplikationen: Das Rückgängig-Machen einer Änderung, die ein anderer Nutzer anschließend modifiziert hat, kann Konflikte erzeugen.

## How It Could Be

> Die Abwesenheit von Rückgängig-Machen in Legacy-Systemen schafft eine Kultur vorsichtiger, langsamer Interaktion, in der Nutzer Angst haben zu erkunden oder zu experimentieren.

Ein von einem Marketingteam genutztes Legacy-Content-Management-System hat keine Rückgängig-Fähigkeit. Jede Textänderung, jeder Bildersatz und jede Layout-Modifikation ist sofort permanent. Teammitglieder haben sich angewöhnt, ganze Seiten in Backup-Ordner zu kopieren, bevor sie irgendwelche Bearbeitungen vornehmen, was das System mit Hunderten von Backup-Kopien überfüllt. Das Team implementiert ein Versionshistoriensystem, das automatisch eine Momentaufnahme vor jeder Speicheroperation erstellt, mit einer Vergleichsansicht, die zeigt, was sich zwischen Versionen geändert hat, und einer Ein-Klick-Wiederherstellungsoption. Das Team fügt auch eine zeitlich begrenzte Rückgängig-Toast-Benachrichtigung hinzu, die nach jeder Speicherung mit "Änderungen gespeichert. [Rückgängig]" für schnelle Umkehrungen erscheint. Das Marketing-Personal hört auf, manuelle Backups zu erstellen, das System wird sauberer, und Teammitglieder berichten, bereitwilliger mit Inhalts- und Layout-Änderungen zu experimentieren, weil sie wissen, dass sie leicht zurücksetzen können.
